def fin1(arr, e):
    iters=0
    low=0
    high=len(arr)-1
    while low <= high:
        iters=iters +1
        mid=(low + high)//2
        mid_value=arr[mid]
        if mid_value == e:
            return mid, iters
        elif mid_value < e:
            low = mid +1
        else:
            high = mid -1
    return -1, iters
def find2(arr, e):
    iters=0

