import heapq


def heap_delete(h:list, i):
    # from https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
    # this is O(logn)
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)