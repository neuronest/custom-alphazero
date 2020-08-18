import os

from graphviz import Digraph


class MctsVisualizer:
    def __init__(self, mcts_root_node, mcts_name="mcts", show_node_index=True):
        self.mcts_root_node = mcts_root_node
        self.mcts_name = mcts_name
        self.show_node_index = show_node_index
        self.node_ref_index = {}
        self.graph_mcts = self.mcts_graph_from_edges(
            MctsVisualizer._breadth_first_edges(self.mcts_root_node),
            remove_unvisited=True,
        )

    @staticmethod
    def _breadth_first_edges(root_node):
        nodes = []
        edges = []
        stack_nodes = [root_node]
        while stack_nodes:
            current_node = stack_nodes.pop(0)
            nodes.append(current_node)
            for edge in current_node.edges:
                child = edge.child
                stack_nodes.append(child)
                edges.append(edge)
        return edges

    def _describe_node(self, node, round_value_at=2):
        if id(node) not in self.node_ref_index:
            self.node_ref_index[id(node)] = len(self.node_ref_index)

        node_description = (
            f"node #{self.node_ref_index[id(node)]}{os.linesep}{os.linesep}"
            if self.show_node_index
            else ""
        )
        node_description += node.board.repr_graphviz()
        node_description += (
            f"{os.linesep}{os.linesep}V={round(node.evaluated_value, round_value_at)}"
            if node.evaluated_value is not None
            else ""
        )
        return node_description

    @staticmethod
    def _describe_edge(edge, round_value_at=2):
        uct = round(edge.upper_confidence_bound(), round_value_at)
        q_value = round(edge.exploitation_term(), round_value_at)
        u = round(edge.exploration_term(), round_value_at)
        p = round(edge.prior, round_value_at)
        n = edge.visit_count
        # .x just when with gravity and for connect_n, find something more general
        label = f"UCT={uct} Q={q_value} U={u}"
        label += f"{os.linesep} P={p} N={n} A={edge.action.x}"
        color = "red" if edge.selected else "black"
        return {"label": label, "color": color}

    def mcts_graph_from_edges(self, edges, remove_unvisited=True):

        if remove_unvisited:
            edges = [edge for edge in edges if edge.visit_count > 0]

        graph_mcts = Digraph("G", filename=f"{self.mcts_name}.gv")
        for edge in edges[:]:
            graph_mcts.edge(
                self._describe_node(edge.parent),
                self._describe_node(edge.child),
                color=MctsVisualizer._describe_edge(edge)["color"],
                label=MctsVisualizer._describe_edge(edge)["label"],
            )
        return graph_mcts

    def save_as_pdf(self, filename=None, directory=None, remove_gv_file=True):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        filename = filename if filename is not None else self.graph_mcts.filename
        # .render outputs 2 files: a .gv file and a .gv.pdf
        self.graph_mcts.render(view=False, filename=filename, directory=directory)
        if remove_gv_file:
            MctsVisualizer._remove_gv(filename, directory=directory)

    @staticmethod
    def _remove_gv(filename, directory=None):
        if directory is None:
            os.remove(filename)
        else:
            os.remove(os.path.join(directory, filename))

    def show(self, filename=None, directory=None):
        """Save the source to file and open the rendered result in a viewer depending on platform
        """
        self.graph_mcts.view(filename=filename, directory=directory)
