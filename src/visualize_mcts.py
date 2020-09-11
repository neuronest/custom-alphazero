import os

from graphviz import Digraph


class MctsVisualizer:
    def __init__(
        self,
        mcts_root_node=None,
        mcts_name="mcts",
        show_node_index=True,
        remove_unplayed_edge=False,
        is_updated=False,
    ):
        self.mcts_root_node = mcts_root_node
        self.mcts_name = mcts_name
        self.show_node_index = show_node_index
        self.remove_unplayed_edge = remove_unplayed_edge
        self.node_ref_index = {}
        self.is_updated = is_updated
        if self.mcts_root_node:
            self.edges = MctsVisualizer._breadth_first_edges(self.mcts_root_node)
            MctsVisualizer._enrich_edges(self.edges)
            self.graph_mcts = self.mcts_graph(remove_unvisited=True)

    def build_mcts_graph(
        self, mcts_root_node, mcts_name=None, remove_unplayed_edge=False
    ):
        self.mcts_root_node = mcts_root_node
        self.mcts_name = mcts_name if mcts_name is not None else self.mcts_name
        self.remove_unplayed_edge = remove_unplayed_edge
        self.edges = MctsVisualizer._breadth_first_edges(self.mcts_root_node)
        MctsVisualizer._enrich_edges(self.edges)
        self.graph_mcts = self.mcts_graph(remove_unvisited=True)

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
    def _describe_edge(edge):
        uct = edge.upper_confidence_bound()
        q_value = edge.exploitation_term()
        u = edge.exploration_term()
        p = edge.prior
        n = edge.visit_count
        p_n = edge.proportion_n
        label = (
            f"UCT={uct:.2f} Q={q_value:.2f} U={u:.2f} {os.linesep} "
            f"P={p:.2f} N={n} PN={p_n:.2f} A={edge.action}"
        )
        color = "red" if edge.played else "black"
        line_width = "4" if edge.greedily_played else "1"
        return {"label": label, "color": color, "line_width": line_width}

    @staticmethod
    def _enrich_edges(edges):
        MctsVisualizer._add_visits_proportions_to_edges(edges)

    @staticmethod
    def _add_visits_proportions_to_edges(edges):
        nodes_analyzed = set()
        for edge in edges:
            node = edge.parent
            if id(node) in nodes_analyzed:
                continue
            else:
                node_edges_all_visits = sum(
                    [node_edge.visit_count for node_edge in node.edges]
                )
                for node_edge in node.edges:
                    # set all proportions to 0 if no edge has been visited from this node
                    node_edge.proportion_n = (
                        float(node_edge.visit_count) / node_edges_all_visits
                        if node_edges_all_visits > 0
                        else 0
                    )
                nodes_analyzed.add(id(node))

    def mcts_graph(self, remove_unvisited=True):
        edges = (
            [edge for edge in self.edges if edge.visit_count > 0]
            if remove_unvisited
            else self.edges
        )
        edges = (
            [edge for edge in edges if edge.played]
            if self.remove_unplayed_edge
            else edges
        )
        graph_mcts = Digraph("G", filename=f"{self.mcts_name}.gv")
        for edge in edges:
            edge_description = MctsVisualizer._describe_edge(edge)
            graph_mcts.edge(
                self._describe_node(edge.parent),
                self._describe_node(edge.child),
                color=edge_description["color"],
                label=edge_description["label"],
                penwidth=edge_description["line_width"],
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
