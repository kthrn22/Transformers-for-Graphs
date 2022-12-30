def trace_shortest_path(root_node, current_node, adjacency_list, distances, max_shortest_len):
    if distances[current_node] == -1 or root_node == current_node:
        return [-1] * max_shortest_len

    trace, shortest_path = [current_node], []
    while trace[-1] != root_node:
        u = trace[-1]
        
        for v, edge_len, edge_id in adjacency_list[u]:
            if distances[u] == distances[v] + edge_len:
                trace.append(v)
                shortest_path.append(edge_id)
                break

    shortest_path += [-1] * (max_shortest_len - len(shortest_path))

    return shortest_path