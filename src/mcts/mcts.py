import numpy as np
from typing import Optional, List, Tuple, Union

from src.config import ConfigGeneral, ConfigMCTS
from src.mcts.utils import normalize_probabilities
from src.model.tensorflow.model import PolicyValueModel
from src.serving.factory import infer_sample

if ConfigGeneral.game == "chess":
    from src.chess.board import Board
    from src.chess.move import Move
elif ConfigGeneral.game == "connect_n":
    from src.connect_n.board import Board
    from src.connect_n.move import Move
else:
    raise NotImplementedError


class UCTEdge:
    def __init__(
        self, parent: "UCTNode", child: "UCTNode", action: Optional[Move], prior: float
    ):
        self.parent = parent
        self.child = child
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_action_value = 0.0

    @property
    def siblings(self):
        return [edge for edge in self.parent.edges if edge is not self]

    def exploitation_term(self) -> float:
        try:
            return self.total_action_value / self.visit_count
        except ZeroDivisionError:
            return 0.0

    def exploration_term(self, override_prior: Optional[float]) -> float:
        prior = override_prior if override_prior is not None else self.prior
        return (
            ConfigMCTS.exploration_constant
            * prior
            * (sum(edge.visit_count for edge in self.parent.edges) ** 0.5)
            / (1 + self.visit_count)
        )

    def upper_confidence_bound(self, override_prior: Optional[float] = None) -> float:
        return self.exploitation_term() + self.exploration_term(override_prior)


class UCTNode:
    def __init__(self, board: Board, edges: List[UCTEdge]):
        self.board = board
        self.edges = edges

    def get_best_edge(self) -> UCTEdge:
        best_child_index = int(
            np.argmax([edge.upper_confidence_bound() for edge in self.edges])
        )
        return self.edges[best_child_index]

    def get_best_edge_with_noise(self) -> UCTEdge:
        priors = np.asarray([edge.prior for edge in self.edges])
        noisy_priors = (
            1 - ConfigMCTS.dirichlet_noise_ratio
        ) * priors + ConfigMCTS.dirichlet_noise_ratio * np.random.dirichlet(
            np.ones(len(priors)) * ConfigMCTS.dirichlet_noise_value
        )
        best_child_index = int(
            np.argmax(
                [
                    edge.upper_confidence_bound(override_prior=noisy_prior)
                    for edge, noisy_prior in zip(self.edges, noisy_priors)
                ]
            )
        )
        return self.edges[best_child_index]


class MCTS:
    def __init__(
        self,
        board: Board,
        all_possible_moves: List[Move],
        concurrency: bool,
        model: Optional[PolicyValueModel] = None,
    ) -> None:
        self.board = board
        self.all_possible_moves = all_possible_moves
        self.concurrency = concurrency
        self.root = self.initialize_root()
        self.path_cache = []
        self.model = model

    def initialize_root(self) -> UCTNode:
        return UCTNode(edges=[], board=self.board)

    def select(self) -> UCTNode:
        current_node = self.root
        while len(current_node.edges):
            if current_node is self.root and ConfigMCTS.enable_dirichlet_noise:
                best_edge = current_node.get_best_edge_with_noise()
            else:
                best_edge = current_node.get_best_edge()
            self.path_cache.append(best_edge)
            current_node = best_edge.child
        return current_node

    def evaluate_and_expand(self, node: UCTNode) -> float:
        if self.model is not None:
            probabilities, value = self.model(
                np.expand_dims(node.board.full_state, axis=0)
            )
            probabilities, value = (
                probabilities.numpy().ravel(),
                value.numpy().item(),
            )
        else:
            probabilities, value = infer_sample(
                node.board.full_state, concurrency=self.concurrency
            )
        probabilities = probabilities[
            node.board.legal_moves_mask(self.all_possible_moves)
        ]
        probabilities = normalize_probabilities(probabilities)
        for prior, move in zip(probabilities, node.board.moves):
            board_child = node.board.play(move, on_copy=True, keep_same_player=True)
            node.edges.append(
                UCTEdge(
                    parent=node,
                    child=UCTNode(board=board_child, edges=[]),
                    action=move,
                    prior=prior,
                )
            )
        return value

    def backup(self, value: float):
        for edge in self.path_cache:
            edge.visit_count += 1
            edge.total_action_value += value
        self.path_cache = []

    def search(self, iterations_number: int):
        for _ in range(iterations_number):
            leaf_node = self.select()
            if leaf_node.board.moves:
                value = self.evaluate_and_expand(leaf_node)
            else:
                value = self.path_cache[-1].exploitation_term()
            self.backup(value)

    def play(
        self,
        greedy: bool = False,
        return_details: bool = False,
        deterministic: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, Move], Board]:
        node = self.root
        if greedy:
            index_max = np.argmax([edge.visit_count for edge in node.edges])
            probabilities = np.zeros(len(node.edges)).astype(float)
            probabilities[index_max] = 1.0
        else:
            probabilities = np.asarray(
                [edge.visit_count for edge in node.edges]
            ).astype(float)
            probabilities = normalize_probabilities(probabilities)
        if deterministic:
            edge = node.edges[np.argmax(probabilities)]
        else:
            edge = np.random.choice(node.edges, 1, p=probabilities).item()
        self.board.play(edge.action, keep_same_player=True)
        self.root = edge.child
        if return_details:
            policy = np.zeros(len(self.all_possible_moves))
            legal_moves_indexes = [
                self.all_possible_moves.index(edge.action) for edge in node.edges
            ]
            policy[legal_moves_indexes] = probabilities
            return (
                self.board.full_state,
                self.board.full_state_mirror,
                policy,
                edge.action,
            )
        else:
            return self.board
