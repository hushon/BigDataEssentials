def bfs(lut, root):
    queue = [root]
    visited = {root}
    new_lut = {}
    level = [{root}]
    while len(queue)>0:
        curr_node = queue.pop()
        # append next_nodes to level that are not visited.
        # mark those nodes in level set as visited.

        level.append(set([n for n in lut[curr_node] if n not in visited]))
        visited.update(set(level[-1]))

        queue.insert(0, n)

        # play with curr_node here.


def bfs(lut, root):
    level_0 = [root]
    level_1 = level_0.flatmap(lambda x: lut[x]).distinct().filter(lambda x: x not in level_0)

