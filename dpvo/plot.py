import matplotlib.pyplot as plt
import networkx as nx

# Manually prepared data for frames n = 0 to n = 10
dpvo_results = [
    {'frame': 0, 'forward_edges': {'ii': [], 'jj': []}, 'backward_edges': {'ii': [], 'jj': []}},
    {'frame': 1, 'forward_edges': {'ii': [], 'jj': []}, 'backward_edges': {'ii': list(range(96)), 'jj': [0]*96}},
    {'frame': 2, 'forward_edges': {'ii': list(range(96)), 'jj': [1]*96}, 'backward_edges': {'ii': list(range(96, 192)), 'jj': [0]*96}},
    {'frame': 3, 'forward_edges': {'ii': list(range(96, 192)), 'jj': [2]*96}, 'backward_edges': {'ii': list(range(192, 288)), 'jj': [0]*96}},
    {'frame': 4, 'forward_edges': {'ii': list(range(192, 288)), 'jj': [3]*96}, 'backward_edges': {'ii': list(range(288, 384)), 'jj': [0]*96}},
    {'frame': 5, 'forward_edges': {'ii': list(range(288, 384)), 'jj': [4]*96}, 'backward_edges': {'ii': list(range(384, 480)), 'jj': [0]*96}},
    {'frame': 6, 'forward_edges': {'ii': list(range(384, 480)), 'jj': [5]*96}, 'backward_edges': {'ii': list(range(480, 576)), 'jj': [0]*96}},
    {'frame': 7, 'forward_edges': {'ii': list(range(480, 576)), 'jj': [6]*96}, 'backward_edges': {'ii': list(range(576, 672)), 'jj': [0]*96}},
    {'frame': 8, 'forward_edges': {'ii': list(range(576, 672)), 'jj': [7]*96}, 'backward_edges': {'ii': list(range(672, 768)), 'jj': [0]*96}},
    {'frame': 9, 'forward_edges': {'ii': list(range(672, 768)), 'jj': [8]*96}, 'backward_edges': {'ii': list(range(768, 864)), 'jj': [0]*96}},
    {'frame': 10, 'forward_edges': {'ii': list(range(768, 864)), 'jj': [9]*96}, 'backward_edges': {'ii': list(range(864, 960)), 'jj': [0]*96}},
]

def plot_graph(dpvo_results):
    G = nx.DiGraph()

    # Add nodes
    for result in dpvo_results:
        frame = result['frame']
        G.add_node(frame, label=f'Frame {frame}')

    # Add edges
    for result in dpvo_results:
        frame = result['frame']
        for ii, jj in zip(result['forward_edges']['ii'], result['forward_edges']['jj']):
            G.add_edge(jj, frame)
        for ii, jj in zip(result['backward_edges']['ii'], result['backward_edges']['jj']):
            G.add_edge(frame, jj)

    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    plt.title('Frame Connectivity Graph from n=0 to n=10')
    plt.show()

# Plot the graph
plot_graph(dpvo_results)