# distutils: language = c++
# cython: language_level=2


Graph<short, short, long long>
#     captype, tcaptype, flowtype
# typedef int node_id
# typedef enum
# 	{
# 		SOURCE	= 0,
# 		SINK	= 1
# 	} termtype;

cdef extern from "fprotocol.h":
  cppclass Graph:
    # Constructor
    Graph(int node_num_max, int edge_num_max, void (*err_function)(char *) = NULL);
    # Attributes
    node_id add_node(int num)
    void add_edge(node_id i, node_id j, captype cap, captype rev_cap)
    void add_tweights(node_id i, tcaptype cap_source, tcaptype cap_sink)
    flowtype maxflow(bool reuse_trees = false, Block<node_id>* changed_list = NULL)
    termtype what_segment(node_id i, termtype default_segm = SOURCE)
