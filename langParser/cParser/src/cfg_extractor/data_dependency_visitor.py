from collections import defaultdict
from src.antlr.gen.CPP14_v2Parser import CPP14_v2Parser
from src.antlr.gen.CPP14_v2Visitor import CPP14_v2Visitor
from src.cfg_extractor.lang_structures import (embed_in_function_structure, embed_in_do_while_structure,
                                               embed_in_for_structure, embed_in_switch_structure,
                                               embed_in_if_structure, embed_in_if_else_structure,
                                               embed_in_while_structure, embed_in_try_catch_structure)
from src.graph.utils import (build_single_node_graph, concat_graphs, build_isolated_node_graph)
from .cfg_extractor_visitor import CFGExtractorVisitor

import networkx as nx
import matplotlib.pyplot as plt

class DataDependencyVisitor(CPP14_v2Visitor):

    def __init__(self):
        super().__init__()
        # Variable table will map each variable to its last write location (context)
        self.variable_table = defaultdict(list)
        # Initialize a directed graph to store data dependencies
        self.data_dependency_graph = nx.DiGraph()

    def add_variable_definition(self, variable, ctx):
        """
        Log the definition (write) of a variable.
        """
        assert(1==0)#print(variable)
        self.variable_table[variable].append(ctx)

    def check_variable_use(self, variable, ctx):
        """
        For every use (read) of a variable, check its last definition and add a data dependency.
        """
        if variable in self.variable_table:
            # For simplicity, we only consider the most recent definition (last write)
            last_write_ctx = self.variable_table[variable][-1]
            
            # Create a data dependency edge in the graph
            self.data_dependency_graph.add_edge(last_write_ctx, ctx)
            
            # Optionally, add additional information, like variable name, to the edge
            self.data_dependency_graph[last_write_ctx][ctx]['variable'] = variable
    # Override or add visit methods to handle variable definitions and uses:

    def visitHandler(self, ctx: CPP14_v2Parser.HandlerContext):
        return self.visit(ctx.compoundstatement())
    
    def visitAssignmentexpression1(self, ctx: CPP14_v2Parser.Assignmentexpression1Context):
        try:
            print('visitAssignmentexpression1')
            #assert(1==0)
            variable = str(ctx.conditionalexpression())  # Simplified: you might need to handle more complex expressions
            #print(dir(variable.getChild))
            variable = variable.getChild
            self.add_variable_definition(variable, ctx)
            return super().visitAssignmentexpression1(ctx)
        except Exception as e:
            print('dir(ctx):')
            print(dir(ctx))
    
    def visitAssignmentexpression2(self, ctx: CPP14_v2Parser.Assignmentexpression2Context):
        try:
            print('visitAssignmentexpression2')
            #assert(1==0)
            # Extract the left-hand side of the assignment
            variable = str(ctx.logicalorexpression())
            
            # Log the variable definition (write)
            self.add_variable_definition(variable, ctx)
            return super().visitAssignmentexpression2(ctx)
        except Exception as e:
            print(dir(ctx))
        
    
    def visitAssignmentexpression3(self, ctx: CPP14_v2Parser.Assignmentexpression3Context):
        try:
            print('visitAssignmentexpression3')
            #assert(1==0) 
            variable = str(ctx.left)  # Simplified: you might need to handle more complex expressions
            self.add_variable_definition(variable, ctx)
            return super().visitAssignmentexpression3(ctx)
        except Exception as e:
            print(dir(ctx))

    def visitExpression(self, ctx: CPP14_v2Parser.ExpressionContext):
        print('visitExpression')
        variable = str(ctx)  # Simplified: you might need to handle more complex expressions
        self.check_variable_use(variable, ctx)
        return super().visitExpression(ctx)
    
    def draw_data_dependency_graph(self, save_path=None):
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.data_dependency_graph)
        nx.draw(self.data_dependency_graph, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=20, font_weight='bold', width=2.5, edge_color='gray')
        edge_labels = nx.get_edge_attributes(self.data_dependency_graph, 'variable')
        nx.draw_networkx_edge_labels(self.data_dependency_graph, pos, edge_labels=edge_labels, font_size=15, font_color='red')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    # Add more methods to handle other types of statements or expressions where variables are defined or used.
