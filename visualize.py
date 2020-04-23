from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import genotypes
from graphviz import Digraph


def plot(genotype_normal, genotype_reduce, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontname="times"),
        node_attr=dict(
            style='filled',
            shape='ellipse',
            align='center',
            height='0.5',
            width='0.5',
            penwidth='2',
            fontname="times"),
        engine='dot')

    g.body.extend(['rankdir=LR'])

    g.node("reduce_c_{k-2}", fillcolor='darkseagreen2')
    g.node("reduce_c_{k-1}", fillcolor='darkseagreen2')
    g.node("normal_c_{k-2}", fillcolor='darkseagreen2')
    g.node("normal_c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype_normal) % 2 == 0
    steps = len(genotype_normal) // 2

    for i in range(steps):
        g.node('n_' + str(i), fillcolor='lightblue')
    for i in range(steps):
        g.node('r_' + str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype_normal[k]
            if j == 0:
                u = "normal_c_{k-2}"
            elif j == 1:
                u = "normal_c_{k-1}"
            else:
                u = 'n_' + str(j - 2)
            v = 'n_' + str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype_reduce[k]
            if j == 0:
                u = "reduce_c_{k-2}"
            elif j == 1:
                u = "reduce_c_{k-1}"
            else:
                u = 'r_' + str(j - 2)
            v = 'r_' + str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("r_c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge('r_' + str(i), "r_c_{k}", fillcolor="gray")
    g.node("n_c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge('n_' + str(i), "n_c_{k}", fillcolor="gray")
    g.render(filename, view=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)
    genotype_name = sys.argv[1]

    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, genotype.reduce, genotype_name)