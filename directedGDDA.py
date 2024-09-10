import networkx as nx
import numpy as np
from tqdm import tqdm



def agreement(G1, G2, method="arithmetic", verbose=False):
    """
    This function computes the GDD-agreement with directed graphlets.

    Arguments
    ----------
    G1, G2 : networkx.DiGraph
        Directed graphs.

    method : str
        Mean method.
          * "arithmetic" : Arithmetic mean.
          * "geometric" : Geometric mean.
          * "vector" : Vector before computing the GDD-agreement.

    verbose : bool
        Show a progress bar.
    """
    # GDD
    if verbose: print("Computing the GDD of the 1st graph.")
    gdd1 = graphlet_degree_distribution(G1, verbose)

    if verbose: print("Computing the GDD of the 2nd graph.")
    gdd2 = graphlet_degree_distribution(G2, verbose)


    # Agreement vector
    a = _agreement(gdd1, gdd2)


    if method == "arithmetic":
        return a.mean()
    elif method == "geometric":
        return np.exp(np.log(a).mean())
    elif method == "vector":
        return a
    else:
        raise ValueError("method must be 'arithmetic', 'geometric' or 'vector'.")



def _agreement(gdd1, gdd2):

    def padded_norm(x, y):
        nx = len(x)
        ny = len(y)

        if nx > ny:
            x_pad = x
            y_pad = np.pad(y, (0, nx - ny))
        elif nx < ny:
            x_pad = np.pad(x, (0, ny - nx))
            y_pad = y
        else:
            x_pad = x
            y_pad = y

        return np.linalg.norm(x_pad - y_pad)


    # Normalization
    gdd1_normalized = gdd_normalize(gdd1)
    gdd2_normalized = gdd_normalize(gdd2)

    # Normalized distances
    distances = np.array([ padded_norm(d1, d2) for d1, d2 in zip(gdd1_normalized, gdd2_normalized) ])
    distances /= np.sqrt(2)

    return 1 - distances



def gdd_normalize(gdd):
    # Scaling
    S = [ np.array([ d / k if k != 0 else 0 for k, d in enumerate(d_j) ]) for d_j in gdd ]

    # Total
    T = [ s_j.sum() for s_j in S ]

    # Normalization
    N = [ s_j / t_j if t_j != 0 else np.array([0.0]) for s_j, t_j in zip(S, T) ]

    return N





def graphlet_degree_distribution(G, verbose=False):
    """
    This function computes the graphlet degree distribution of graph G.

    Arguments
    ----------
    G : networkx.Graph
        Undirected graph.

    verbose : bool
        Show a progress bar.
    """
    # Feature matrix (n_nodes x n_orbits)
    F = orbital_features(G, verbose)
    F = np.array([ x for x in F.values() ])

    return [ np.bincount(F[:, i]) for i in range(F.shape[1]) ]





def graphlet_decomposition(G, verbose=False):
    """
    This function counts the number of graphlets in graph G.

    Arguments
    ----------
    G : networkx.Graph
        Undirected graph.

    verbose : bool
        Show a progress bar.
    """
    # Representative orbits for graphlets
    representative_orbits = [0, 2, 6, 8, 9, 10, 13, 17, 21, 25,
                             30, 32, 33, 36, 39, 40, 44, 47,
                             49, 53, 57, 61, 65, 69, 73, 77,
                             81, 85, 88, 91, 94, 97, 101, 105, 109, 114,
                             118, 120, 121, 125]

    # GDD
    gdd = graphlet_degree_distribution(G, verbose)

    # Counts for graphlets
    counts = np.array([ np.dot(d, np.arange(len(d))) for d in gdd ])
    counts = counts[representative_orbits]

    # Correction
    counts[4]  /= 3
    counts[14] /= 4
    counts[16] /= 2

    return counts





def orbital_features(G, verbose=False):
    """
    This function computes the orbital features from graph G.

    Arguments
    ----------
    G : networkx.Graph
        Undirected graph.

    verbose : bool
        Show a progress bar.
    """
    A = nx.to_scipy_sparse_array(G, format="lil")
    F = _orbital_features(A, verbose)

    return dict(zip(G.nodes(), F))



def _orbital_features(A, verbose=False):

    def comb(n):
        return n * (n - 1) // 2

    # Number of orbits in 2--4 nodes graphlets
    n_orbits = 129

    # Graph
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)

    # Undirected graph
    G_undirected = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    # Numbers of nodes and edges
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Feature matrix (n_nodes x n_orbits)
    F = np.zeros((n_nodes, n_orbits), dtype=np.int64)

    # Progress bar
    if verbose:
        edge_list = tqdm(G.edges(), total=n_edges)
    else:
        edge_list = G.edges()


    # Orbits 0, 1 (outdegree, indegree)
    F[:, 0] = [ d for _, d in G.out_degree() ]
    F[:, 1] = [ d for _, d in G.in_degree()  ]


    # Neighbors
    N_out = { v: set(G.successors(v))   for v in G.nodes() }
    N_in  = { v: set(G.predecessors(v)) for v in G.nodes() }


    # Orbits 2--128

    for u, v in edge_list:

        e = {u, v}

        Nu_out = N_out[u] - {v}
        Nu_in  = N_in[u]  - {v}
        Nu = Nu_out | Nu_in

        Nv_out = N_out[v] - {u}
        Nv_in  = N_in[v]  - {u}
        Nv = Nv_out | Nv_in

        Ne = Nu | Nv | e

        Star_u_out = Nu_out - Nv
        Star_u_in  = Nu_in  - Nv
        Star_v_out = Nv_out - Nu
        Star_v_in  = Nv_in  - Nu

        Tri_e_io = Nu_in  & Nv_out
        Tri_e_oo = Nu_out & Nv_out
        Tri_e_oi = Nu_out & Nv_in
        Tri_e_ii = Nu_in  & Nv_in

        n_Su_out = len(Star_u_out)
        n_Su_in  = len(Star_u_in)
        n_Sv_out = len(Star_v_out)
        n_Sv_in  = len(Star_v_in)

        n_Te_io = len(Tri_e_io)
        n_Te_oo = len(Tri_e_oo)
        n_Te_oi = len(Tri_e_oi)
        n_Te_ii = len(Tri_e_ii)


        # 3-node graphlets

        n_2_3 = n_Sv_out
        n_3_4 = n_Su_in
        n_6_5 = n_Su_out
        n_7_8 = n_Sv_in

        n_9_9   = n_Te_io
        n_11_10 = n_Te_oi
        n_11_12 = n_Te_oo
        n_12_10 = n_Te_ii


        # 4-node graphlets

        n_13_14 = 0
        #n_14_15 = 0
        n_15_16 = 0
        n_17_18 = 0
        #n_18_19 = 0
        n_20_19 = 0
        n_21_22 = 0
        #n_23_22 = 0
        n_23_24 = 0
        n_26_25 = 0
        #n_27_26 = 0
        n_27_28 = 0
        #n_29_30 = 0
        #n_32_31 = 0
        #n_33_34 = 0
        #n_34_35 = 0
        #n_37_36 = 0
        #n_38_37 = 0
        n_39_39 = 0
        n_40_41 = 0
        n_41_43 = 0
        n_42_40 = 0
        n_42_43 = 0
        n_45_44 = 0
        n_46_47 = 0
        n_48_46 = 0
        n_49_50 = 0
        #n_50_51 = 0
        #n_51_49 = 0
        n_51_52 = 0
        n_53_54 = 0
        #n_54_55 = 0
        #n_55_53 = 0
        n_56_55 = 0
        n_57_58 = 0
        #n_57_59 = 0
        #n_58_59 = 0
        n_59_60 = 0
        n_61_62 = 0
        #n_61_63 = 0
        #n_62_63 = 0
        n_64_63 = 0
        n_65_66 = 0
        #n_67_65 = 0
        #n_67_66 = 0
        n_67_68 = 0
        n_69_70 = 0
        #n_71_69 = 0
        #n_71_70 = 0
        n_72_71 = 0
        n_73_74 = 0
        #n_73_75 = 0
        #n_75_74 = 0
        n_75_76 = 0
        n_77_78 = 0
        #n_77_79 = 0
        #n_79_78 = 0
        n_80_79 = 0
        n_81_83 = 0
        n_82_81 = 0
        n_83_84 = 0
        #n_84_81 = 0
        n_84_82 = 0
        n_86_85 = 0
        #n_87_85 = 0
        n_87_86 = 0
        #n_88_90 = 0
        n_89_88 = 0
        n_90_89 = 0
        n_92_91 = 0
        n_92_93 = 0
        #n_93_91 = 0
        n_94_95 = 0
        #n_96_94 = 0
        n_96_95 = 0
        n_97_99 = 0
        n_98_97 = 0
        #n_100_97  = 0
        n_100_98  = 0
        n_100_99  = 0
        n_101_103 = 0
        #n_101_104 = 0
        n_102_101 = 0
        n_104_102 = 0
        n_104_103 = 0
        n_106_105 = 0
        n_107_105 = 0
        n_107_108 = 0
        #n_108_105 = 0
        n_108_106 = 0
        #n_109_112 = 0
        n_110_109 = 0
        n_111_109 = 0
        n_111_112 = 0
        n_112_110 = 0
        n_113_114 = 0
        n_113_116 = 0
        n_114_115 = 0
        #n_114_116 = 0
        n_116_115 = 0
        n_117_117 = 0
        n_117_118 = 0
        n_119_119 = 0
        n_120_119 = 0
        n_121_123 = 0
        n_121_124 = 0
        n_122_121 = 0
        n_123_122 = 0
        n_124_122 = 0
        n_124_123 = 0
        n_126_125 = 0
        n_126_127 = 0
        n_127_125 = 0
        n_128_125 = 0
        n_128_126 = 0
        n_128_127 = 0


        for w in Star_u_out:

            Nw_out = N_out[w] - e

            n_27_28   += len(Nw_out - Ne)
            n_67_68   += len(Nw_out & Star_u_out)
            n_51_52   += len(Nw_out & Star_u_in)
            n_48_46   += len(Nw_out & Star_v_out)
            n_42_43   += len(Nw_out & Star_v_in)
            n_90_89   += len(Nw_out & Tri_e_io)
            n_87_86   += len(Nw_out & Tri_e_oo)
            n_100_99  += len(Nw_out & Tri_e_oi)
            n_104_103 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_23_24   += len(Nw_in - Ne)
            #n_67_68   += len(Nw_in & Star_u_out)
            n_75_76   += len(Nw_in & Star_u_in)
            n_42_40   += len(Nw_in & Star_v_out)
            n_45_44   += len(Nw_in & Star_v_in)
            n_104_102 += len(Nw_in & Tri_e_io)
            n_100_98  += len(Nw_in & Tri_e_oo)
            n_96_95   += len(Nw_in & Tri_e_oi)
            n_94_95   += len(Nw_in & Tri_e_ii)


        for w in Star_u_in:

            Nw_out = N_out[w] - e

            n_26_25   += len(Nw_out - Ne)
            #n_75_76   += len(Nw_out & Star_u_out)
            n_59_60   += len(Nw_out & Star_u_in)
            n_40_41   += len(Nw_out & Star_v_out)
            n_46_47   += len(Nw_out & Star_v_in)
            n_112_110 += len(Nw_out & Tri_e_io)
            n_108_106 += len(Nw_out & Tri_e_oo)
            n_114_115 += len(Nw_out & Tri_e_oi)
            n_116_115 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_15_16   += len(Nw_in - Ne)
            #n_51_52   += len(Nw_in & Star_u_out)
            #n_59_60   += len(Nw_in & Star_u_in)
            n_39_39   += len(Nw_in & Star_v_out)
            n_41_43   += len(Nw_in & Star_v_in)
            n_81_83   += len(Nw_in & Tri_e_io)
            n_84_82   += len(Nw_in & Tri_e_oo)
            n_101_103 += len(Nw_in & Tri_e_oi)
            n_97_99   += len(Nw_in & Tri_e_ii)


        for w in Star_v_out:

            Nw_out = N_out[w] - e

            n_13_14   += len(Nw_out - Ne)
            #n_42_40   += len(Nw_out & Star_u_out)
            #n_39_39   += len(Nw_out & Star_u_in)
            n_72_71   += len(Nw_out & Star_v_out)
            n_56_55   += len(Nw_out & Star_v_in)
            n_83_84   += len(Nw_out & Tri_e_io)
            n_107_108 += len(Nw_out & Tri_e_oo)
            n_111_112 += len(Nw_out & Tri_e_oi)
            n_82_81   += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_17_18   += len(Nw_in - Ne)
            #n_48_46   += len(Nw_in & Star_u_out)
            #n_40_41   += len(Nw_in & Star_u_in)
            #n_72_71   += len(Nw_in & Star_v_out)
            n_80_79   += len(Nw_in & Star_v_in)
            n_102_101 += len(Nw_in & Tri_e_io)
            n_113_114 += len(Nw_in & Tri_e_oo)
            n_113_116 += len(Nw_in & Tri_e_oi)
            n_98_97   += len(Nw_in & Tri_e_ii)


        for w in Star_v_in:

            Nw_out = N_out[w] - e

            n_21_22   += len(Nw_out - Ne)
            #n_45_44   += len(Nw_out & Star_u_out)
            #n_41_43   += len(Nw_out & Star_u_in)
            #n_80_79   += len(Nw_out & Star_v_out)
            n_64_63   += len(Nw_out & Star_v_in)
            n_110_109 += len(Nw_out & Tri_e_io)
            n_92_93   += len(Nw_out & Tri_e_oo)
            n_92_91   += len(Nw_out & Tri_e_oi)
            n_106_105 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_20_19   += len(Nw_in - Ne)
            #n_42_43   += len(Nw_in & Star_u_out)
            #n_46_47   += len(Nw_in & Star_u_in)
            #n_56_55   += len(Nw_in & Star_v_out)
            #n_64_63   += len(Nw_in & Star_v_in)
            n_89_88   += len(Nw_in & Tri_e_io)
            n_111_109 += len(Nw_in & Tri_e_oo)
            n_107_105 += len(Nw_in & Tri_e_oi)
            n_86_85   += len(Nw_in & Tri_e_ii)


        for w in Tri_e_io:

            Nw_out = N_out[w] - e

            n_49_50 += len(Nw_out - Ne)
            #n_104_102 += len(Nw_out & Star_u_out)
            #n_81_83 += len(Nw_out & Star_u_in)
            #n_102_101 += len(Nw_out & Star_v_out)
            #n_89_88 += len(Nw_out & Star_v_in)
            n_122_121 += len(Nw_out & Tri_e_io)
            n_117_117 += len(Nw_out & Tri_e_oo)
            n_124_122 += len(Nw_out & Tri_e_oi)
            n_123_122 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_53_54   += len(Nw_in - Ne)
            #n_90_89   += len(Nw_in & Star_u_out)
            #n_112_110 += len(Nw_in & Star_u_in)
            #n_83_84   += len(Nw_in & Star_v_out)
            #n_110_109 += len(Nw_in & Star_v_in)
            #n_122_121 += len(Nw_in & Tri_e_io)
            n_121_124 += len(Nw_in & Tri_e_oo)
            n_121_123 += len(Nw_in & Tri_e_oi)
            n_119_119 += len(Nw_in & Tri_e_ii)


        for w in Tri_e_oo:

            Nw_out = N_out[w] - e

            n_57_58 += len(Nw_out - Ne)
            #n_100_98  += len(Nw_out & Star_u_out)
            #n_84_82   += len(Nw_out & Star_u_in)
            #n_113_114 += len(Nw_out & Star_v_out)
            #n_111_109 += len(Nw_out & Star_v_in)
            #n_121_124 += len(Nw_out & Tri_e_io)
            n_128_126 += len(Nw_out & Tri_e_oo)
            n_120_119 += len(Nw_out & Tri_e_oi)
            n_124_123 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_61_62 += len(Nw_in - Ne)
            #n_87_86   += len(Nw_in & Star_u_out)
            #n_108_106 += len(Nw_in & Star_u_in)
            #n_107_108 += len(Nw_in & Star_v_out)
            #n_92_93   += len(Nw_in & Star_v_in)
            #n_117_117 += len(Nw_in & Tri_e_io)
            #n_128_126 += len(Nw_in & Tri_e_oo)
            n_128_127 += len(Nw_in & Tri_e_oi)
            n_126_127 += len(Nw_in & Tri_e_ii)


        for w in Tri_e_oi:

            Nw_out = N_out[w] - e

            n_73_74   += len(Nw_out - Ne)
            #n_96_95   += len(Nw_out & Star_u_out)
            #n_101_103 += len(Nw_out & Star_u_in)
            #n_113_116 += len(Nw_out & Star_v_out)
            #n_107_105 += len(Nw_out & Star_v_in)
            #n_121_123 += len(Nw_out & Tri_e_io)
            #n_128_127 += len(Nw_out & Tri_e_oo)
            n_128_125 += len(Nw_out & Tri_e_oi)
            n_117_118 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_77_78   += len(Nw_in - Ne)
            #n_100_99  += len(Nw_in & Star_u_out)
            #n_114_115 += len(Nw_in & Star_u_in)
            #n_111_112 += len(Nw_in & Star_v_out)
            #n_92_91   += len(Nw_in & Star_v_in)
            #n_124_122 += len(Nw_in & Tri_e_io)
            #n_120_119 += len(Nw_in & Tri_e_oo)
            #n_128_125 += len(Nw_in & Tri_e_oi)
            n_126_125 += len(Nw_in & Tri_e_ii)


        for w in Tri_e_ii:

            Nw_out = N_out[w] - e

            n_65_66   += len(Nw_out - Ne)
            #n_94_95   += len(Nw_out & Star_u_out)
            #n_97_99   += len(Nw_out & Star_u_in)
            #n_98_97   += len(Nw_out & Star_v_out)
            #n_86_85   += len(Nw_out & Star_v_in)
            #n_119_119 += len(Nw_out & Tri_e_io)
            #n_126_127 += len(Nw_out & Tri_e_oo)
            #n_126_125 += len(Nw_out & Tri_e_oi)
            n_127_125 += len(Nw_out & Tri_e_ii)

            Nw_in = N_in[w] - e

            n_69_70   += len(Nw_in - Ne)
            #n_104_103 += len(Nw_in & Star_u_out)
            #n_116_115 += len(Nw_in & Star_u_in)
            #n_82_81   += len(Nw_in & Star_v_out)
            #n_106_105 += len(Nw_in & Star_v_in)
            #n_123_122 += len(Nw_in & Tri_e_io)
            #n_124_123 += len(Nw_in & Tri_e_oo)
            #n_117_118 += len(Nw_in & Tri_e_oi)
            #n_127_125 += len(Nw_in & Tri_e_ii)



        m_SuSu_oo = 0
        m_SuSu_oi = 0
        m_SuSu_ii = 0
        m_SvSv_oo = 0
        m_SvSv_oi = 0
        m_SvSv_ii = 0
        m_TeTe_io = 0
        m_TeTe_oo = 0
        m_TeTe_oi = 0
        m_TeTe_ii = 0

        m_SuSv_oo = 0
        m_SuSv_oi = 0
        m_SuSv_io = 0
        m_SuSv_ii = 0

        m_SuTe_oio = 0
        m_SuTe_ooo = 0
        m_SuTe_ooi = 0
        m_SuTe_oii = 0
        m_SuTe_iio = 0
        m_SuTe_ioo = 0
        m_SuTe_ioi = 0
        m_SuTe_iii = 0

        m_SvTe_oio = 0
        m_SvTe_ooo = 0
        m_SvTe_ooi = 0
        m_SvTe_oii = 0
        m_SvTe_iio = 0
        m_SvTe_ioo = 0
        m_SvTe_ioi = 0
        m_SvTe_iii = 0

        m_TeTe_iooo = 0
        m_TeTe_iooi = 0
        m_TeTe_ioii = 0
        m_TeTe_oooi = 0
        m_TeTe_ooii = 0
        m_TeTe_oiii = 0


        for s, t in G_undirected.edges():

            if s in Star_u_out:

                if t in Star_u_out:
                    m_SuSu_oo += 1
                elif t in Star_u_in:
                    m_SuSu_oi += 1
                elif t in Star_v_out:
                    m_SuSv_oo += 1
                elif t in Star_v_in:
                    m_SuSv_oi += 1
                elif t in Tri_e_io:
                    m_SuTe_oio += 1
                elif t in Tri_e_oo:
                    m_SuTe_ooo += 1
                elif t in Tri_e_oi:
                    m_SuTe_ooi += 1
                elif t in Tri_e_ii:
                    m_SuTe_oii += 1

            elif s in Star_u_in:

                if t in Star_u_out:
                    m_SuSu_oi += 1
                elif t in Star_u_in:
                    m_SuSu_ii += 1
                elif t in Star_v_out:
                    m_SuSv_io += 1
                elif t in Star_v_in:
                    m_SuSv_ii += 1
                elif t in Tri_e_io:
                    m_SuTe_iio += 1
                elif t in Tri_e_oo:
                    m_SuTe_ioo += 1
                elif t in Tri_e_oi:
                    m_SuTe_ioi += 1
                elif t in Tri_e_ii:
                    m_SuTe_iii += 1

            elif s in Star_v_out:

                if t in Star_u_out:
                    m_SuSv_oo += 1
                elif t in Star_u_in:
                    m_SuSv_io += 1
                elif t in Star_v_out:
                    m_SvSv_oo += 1
                elif t in Star_v_in:
                    m_SvSv_oi += 1
                elif t in Tri_e_io:
                    m_SvTe_oio += 1
                elif t in Tri_e_oo:
                    m_SvTe_ooo += 1
                elif t in Tri_e_oi:
                    m_SvTe_ooi += 1
                elif t in Tri_e_ii:
                    m_SvTe_oii += 1

            elif s in Star_v_in:

                if t in Star_u_out:
                    m_SuSv_oi += 1
                elif t in Star_u_in:
                    m_SuSv_ii += 1
                elif t in Star_v_out:
                    m_SvSv_oi += 1
                elif t in Star_v_in:
                    m_SvSv_ii += 1
                elif t in Tri_e_io:
                    m_SvTe_iio += 1
                elif t in Tri_e_oo:
                    m_SvTe_ioo += 1
                elif t in Tri_e_oi:
                    m_SvTe_ioi += 1
                elif t in Tri_e_ii:
                    m_SvTe_iii += 1

            elif s in Tri_e_io:

                if t in Star_u_out:
                    m_SuTe_oio += 1
                elif t in Star_u_in:
                    m_SuTe_iio += 1
                elif t in Star_v_out:
                    m_SvTe_oio += 1
                elif t in Star_v_in:
                    m_SvTe_oio += 1
                elif t in Tri_e_io:
                    m_TeTe_io += 1
                elif t in Tri_e_oo:
                    m_TeTe_iooo += 1
                elif t in Tri_e_oi:
                    m_TeTe_iooi += 1
                elif t in Tri_e_ii:
                    m_TeTe_ioii += 1

            elif s in Tri_e_oo:

                if t in Star_u_out:
                    m_SuTe_ooo += 1
                elif t in Star_u_in:
                    m_SuTe_ioo += 1
                elif t in Star_v_out:
                    m_SvTe_ooo += 1
                elif t in Star_v_in:
                    m_SvTe_ooo += 1
                elif t in Tri_e_io:
                    m_TeTe_iooo += 1
                elif t in Tri_e_oo:
                    m_TeTe_oo += 1
                elif t in Tri_e_oi:
                    m_TeTe_oooi += 1
                elif t in Tri_e_ii:
                    m_TeTe_ooii += 1

            elif s in Tri_e_oi:

                if t in Star_u_out:
                    m_SuTe_ooi += 1
                elif t in Star_u_in:
                    m_SuTe_ioi += 1
                elif t in Star_v_out:
                    m_SvTe_ooi += 1
                elif t in Star_v_in:
                    m_SvTe_ooi += 1
                elif t in Tri_e_io:
                    m_TeTe_iooi += 1
                elif t in Tri_e_oo:
                    m_TeTe_oooi += 1
                elif t in Tri_e_oi:
                    m_TeTe_oi += 1
                elif t in Tri_e_ii:
                    m_TeTe_oiii += 1

            elif s in Tri_e_ii:

                if t in Star_u_out:
                    m_SuTe_oii += 1
                elif t in Star_u_in:
                    m_SuTe_iii += 1
                elif t in Star_v_out:
                    m_SvTe_oii += 1
                elif t in Star_v_in:
                    m_SvTe_oii += 1
                elif t in Tri_e_io:
                    m_TeTe_ioii += 1
                elif t in Tri_e_oo:
                    m_TeTe_ooii += 1
                elif t in Tri_e_oi:
                    m_TeTe_oiii += 1
                elif t in Tri_e_ii:
                    m_TeTe_ii += 1


        n_32_31 = comb(n_Su_out) - m_SuSu_oo
        n_37_36 = comb(n_Su_in)  - m_SuSu_ii
        n_34_35 = n_Su_out * n_Su_in - len(Star_u_out & Star_u_in) - m_SuSu_oi

        n_33_34 = comb(n_Sv_out) - m_SvSv_oo
        n_29_30 = comb(n_Sv_in)  - m_SvSv_ii
        n_38_37 = n_Sv_out * n_Sv_in - len(Star_v_out & Star_v_in) - m_SvSv_oi

        n_27_26 = n_Su_out * n_Sv_out - m_SuSv_oo
        n_23_22 = n_Su_out * n_Sv_in  - m_SuSv_oi
        n_14_15 = n_Su_in  * n_Sv_out - m_SuSv_io
        n_18_19 = n_Su_in  * n_Sv_in  - m_SuSv_ii

        n_51_49 = n_Su_out * n_Te_io - m_SuTe_oio
        n_67_65 = n_Su_out * n_Te_oo - m_SuTe_ooo
        n_67_66 = n_Su_out * n_Te_oi - m_SuTe_ooi
        n_75_74 = n_Su_out * n_Te_ii - m_SuTe_oii
        n_55_53 = n_Su_in  * n_Te_io - m_SuTe_iio
        n_71_69 = n_Su_in  * n_Te_oo - m_SuTe_ioo
        n_71_70 = n_Su_in  * n_Te_oi - m_SuTe_ioi
        n_79_78 = n_Su_in  * n_Te_ii - m_SuTe_iii

        n_50_51 = n_Sv_out * n_Te_io - m_SvTe_oio
        n_73_75 = n_Sv_out * n_Te_oo - m_SvTe_ooo
        n_57_59 = n_Sv_out * n_Te_oi - m_SvTe_ooi
        n_58_59 = n_Sv_out * n_Te_ii - m_SvTe_oii
        n_54_55 = n_Sv_in  * n_Te_io - m_SvTe_iio
        n_77_79 = n_Sv_in  * n_Te_oo - m_SvTe_ioo
        n_61_63 = n_Sv_in  * n_Te_oi - m_SvTe_ioi
        n_62_63 = n_Sv_in  * n_Te_ii - m_SvTe_iii

        n_88_90 = comb(n_Te_io) - m_TeTe_io
        n_96_94 = comb(n_Te_oo) - m_TeTe_oo
        n_87_85 = comb(n_Te_oi) - m_TeTe_oi
        n_93_91 = comb(n_Te_ii) - m_TeTe_ii

        n_101_104 = n_Te_io * n_Te_oo - len(Tri_e_io & Tri_e_oo) - m_TeTe_iooo
        n_84_81   = n_Te_io * n_Te_oi - len(Tri_e_io & Tri_e_oi) - m_TeTe_iooi
        n_109_112 = n_Te_io * n_Te_ii - len(Tri_e_io & Tri_e_ii) - m_TeTe_ioii
        n_100_97  = n_Te_oo * n_Te_oi - len(Tri_e_oo & Tri_e_oi) - m_TeTe_oooi
        n_114_116 = n_Te_oo * n_Te_ii - len(Tri_e_oo & Tri_e_ii) - m_TeTe_ooii
        n_108_105 = n_Te_oi * n_Te_ii - len(Tri_e_oi & Tri_e_ii) - m_TeTe_iooo


        # Counting

        F[u, 2] += n_2_3
        F[u, 3] += n_3_4
        F[v, 3] += n_2_3
        F[v, 4] += n_3_4
        F[v, 5] += n_6_5
        F[u, 6] += n_6_5
        F[u, 7] += n_7_8
        F[v, 8] += n_7_8

        F[u,  9] += n_9_9
        F[v,  9] += n_9_9
        F[v, 10] += n_11_10 + n_12_10
        F[u, 11] += n_11_10 + n_11_12
        F[u, 12] += n_12_10
        F[v, 12] += n_11_12

        F[u, 13] += n_13_14
        F[u, 14] += n_14_15
        F[v, 14] += n_13_14
        F[u, 15] += n_15_16
        F[v, 15] += n_14_15
        F[v, 16] += n_15_16
        F[u, 17] += n_17_18
        F[u, 18] += n_18_19
        F[v, 18] += n_17_18
        F[v, 19] += n_18_19 + n_20_19
        F[u, 20] += n_20_19
        F[u, 21] += n_21_22
        F[v, 22] += n_21_22 + n_23_22
        F[u, 23] += n_23_22 + n_23_24
        F[v, 24] += n_23_24
        F[v, 25] += n_26_25
        F[u, 26] += n_26_25
        F[v, 26] += n_27_26
        F[u, 27] += n_27_26 + n_27_28
        F[v, 28] += n_27_28

        F[u, 29] += n_29_30
        F[v, 30] += n_29_30
        F[v, 31] += n_32_31
        F[u, 32] += n_32_31
        F[u, 33] += n_33_34
        F[u, 34] += n_34_35
        F[v, 34] += n_33_34
        F[v, 35] += n_34_35
        F[v, 36] += n_37_36
        F[u, 37] += n_37_36
        F[v, 37] += n_38_37
        F[u, 38] += n_38_37

        F[u, 39] += n_39_39
        F[v, 39] += n_39_39
        F[u, 40] += n_40_41
        F[v, 40] += n_42_40
        F[u, 41] += n_41_43
        F[v, 41] += n_40_41
        F[u, 42] += n_42_40 + n_42_43
        F[v, 43] += n_41_43 + n_42_43
        F[v, 44] += n_45_44
        F[u, 45] += n_45_44
        F[u, 46] += n_46_47
        F[v, 46] += n_48_46
        F[v, 47] += n_46_47
        F[u, 48] += n_48_46

        F[u, 49] += n_49_50
        F[v, 49] += n_51_49
        F[u, 50] += n_50_51
        F[v, 50] += n_49_50
        F[u, 51] += n_51_49 + n_51_52
        F[v, 51] += n_50_51
        F[v, 52] += n_51_52
        F[u, 53] += n_53_54
        F[v, 53] += n_55_53
        F[u, 54] += n_54_55
        F[v, 54] += n_53_54
        F[u, 55] += n_55_53
        F[v, 55] += n_54_55 + n_56_55
        F[u, 56] += n_56_55
        F[u, 57] += n_57_58 + n_57_59
        F[u, 58] += n_58_59
        F[v, 58] += n_57_58
        F[u, 59] += n_59_60
        F[v, 59] += n_57_59 + n_58_59
        F[v, 60] += n_59_60
        F[u, 61] += n_61_62 + n_61_63
        F[u, 62] += n_62_63
        F[v, 62] += n_61_62
        F[v, 63] += n_61_63 + n_62_63 + n_64_63
        F[u, 64] += n_64_63
        F[u, 65] += n_65_66
        F[v, 65] += n_67_65
        F[v, 66] += n_65_66 + n_67_66
        F[u, 67] += n_67_65 + n_67_66 + n_67_68
        F[v, 68] += n_67_68
        F[u, 69] += n_69_70
        F[v, 69] += n_71_69
        F[v, 70] += n_69_70 + n_71_70
        F[u, 71] += n_71_69 + n_71_70
        F[v, 71] += n_72_71
        F[u, 72] += n_72_71
        F[u, 73] += n_73_74 + n_73_75
        F[v, 74] += n_73_74 + n_75_74
        F[u, 75] += n_75_74 + n_75_76
        F[v, 75] += n_73_75
        F[v, 76] += n_75_76
        F[u, 77] += n_77_78 + n_77_79
        F[v, 78] += n_77_78 + n_79_78
        F[u, 79] += n_79_78
        F[v, 79] += n_77_79 + n_80_79
        F[u, 80] += n_80_79

        F[u,  81] += n_81_83
        F[v,  81] += n_82_81 + n_84_81
        F[u,  82] += n_82_81
        F[v,  82] += n_84_82
        F[u,  83] += n_83_84
        F[v,  83] += n_81_83
        F[u,  84] += n_84_81 + n_84_82
        F[v,  84] += n_83_84
        F[v,  85] += n_86_85 + n_87_85
        F[u,  86] += n_86_85
        F[v,  86] += n_87_86
        F[u,  87] += n_87_85 + n_87_86
        F[u,  88] += n_88_90
        F[v,  88] += n_89_88
        F[u,  89] += n_89_88
        F[v,  89] += n_90_89
        F[u,  90] += n_90_89
        F[v,  90] += n_88_90
        F[v,  91] += n_92_91 + n_93_91
        F[u,  92] += n_92_91 + n_92_93
        F[u,  93] += n_93_91
        F[v,  93] += n_92_93
        F[u,  94] += n_94_95
        F[v,  94] += n_96_94
        F[v,  95] += n_94_95 + n_96_95
        F[u,  96] += n_96_94 + n_96_95
        F[u,  97] += n_97_99
        F[v,  97] += n_98_97 + n_100_97
        F[u,  98] += n_98_97
        F[v,  98] += n_100_98
        F[v,  99] += n_97_99 + n_100_99
        F[u, 100] += n_100_97 + n_100_98 + n_100_99
        F[u, 101] += n_101_103 + n_101_104
        F[v, 101] += n_102_101
        F[u, 102] += n_102_101
        F[v, 102] += n_104_102
        F[v, 103] += n_101_103 + n_104_103
        F[u, 104] += n_104_102 + n_104_103
        F[v, 104] += n_101_104
        F[v, 105] += n_106_105 + n_107_105 + n_108_105
        F[u, 106] += n_106_105
        F[v, 106] += n_108_106
        F[u, 107] += n_107_105 + n_107_108
        F[u, 108] += n_108_105 + n_108_106
        F[v, 108] += n_107_108
        F[u, 109] += n_109_112
        F[v, 109] += n_110_109 + n_111_109
        F[u, 110] += n_110_109
        F[v, 110] += n_112_110
        F[u, 111] += n_111_109 + n_111_112
        F[u, 112] += n_112_110
        F[v, 112] += n_109_112 + n_111_112
        F[u, 113] += n_113_114 + n_113_116
        F[u, 114] += n_114_115 + n_114_116
        F[v, 114] += n_113_114
        F[v, 115] += n_114_115 + n_116_115
        F[u, 116] += n_116_115
        F[v, 116] += n_113_116 + n_114_116

        F[u, 117] += n_117_117 + n_117_118
        F[v, 117] += n_117_117
        F[v, 118] += n_117_118
        F[u, 119] += n_119_119
        F[v, 119] += n_119_119 + n_120_119
        F[u, 120] += n_120_119
        F[u, 121] += n_121_123 + n_121_124
        F[v, 121] += n_122_121
        F[u, 122] += n_122_121
        F[v, 122] += n_123_122 + n_124_122
        F[u, 123] += n_123_122
        F[v, 123] += n_121_123 + n_124_123
        F[u, 124] += n_124_122 + n_124_123
        F[v, 124] += n_121_124
        F[v, 125] += n_126_125 + n_127_125 + n_128_125
        F[u, 126] += n_126_125 + n_126_127
        F[v, 126] += n_128_126
        F[u, 127] += n_127_125
        F[v, 127] += n_126_127 + n_128_127
        F[u, 128] += n_128_125 + n_128_126 + n_128_127



    # Orbits with degree 2
    orbits = [3, 6, 8, 9, 10, 11, 12,
              14, 15, 18, 19, 22, 23, 26, 27,
              39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
              49, 50, 53, 54, 57, 58, 61, 62, 65, 66, 69, 70, 73, 74, 77, 78,
              82,  83,  89,  92,  95, 98, 99, 102, 103, 106, 107, 110, 111, 113, 115]

    F[:, orbits] //= 2

    # Orbits with degree 3
    orbits = [30, 32, 34, 37, 
              51, 55, 59, 63, 67, 71, 75, 79,
              81, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 100, 101, 104, 105, 108, 109, 112, 114, 116,
              117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]

    F[:, orbits] //= 3


    return F
