"""
Quick Sort Algorithm — Three Implementations
=============================================
Three implementations in increasing sophistication:
  1. Basic version  — simplest implementation (easy to understand but not space-efficient)
  2. In-place partition — standard implementation (commonly asked in interviews)
  3. Random pivot     — production-grade implementation (avoids worst-case behavior)
"""

import random
from typing import List


# ============================================================
# Version 1: Basic (most intuitive, but not in-place)
# Approach: put elements less than / greater than pivot into two new lists
# Pros: very easy to understand; great for beginners
# Cons: requires O(n) extra space
# ============================================================
def quicksort_basic(arr: List[int]) -> List[int]:
    """Quick sort — basic version (simplest implementation)."""
    # Empty array or single element — return as-is
    if len(arr) <= 1:
        return arr

    # Choose the first element as the pivot
    pivot = arr[0]

    # Partition: split all elements into three groups
    left = [x for x in arr[1:] if x < pivot]     # less than pivot
    middle = [x for x in arr if x == pivot]       # equal to pivot
    right = [x for x in arr[1:] if x > pivot]     # greater than pivot

    # Recurse + merge
    return quicksort_basic(left) + middle + quicksort_basic(right)


# ============================================================
# Version 2: In-place partition
# No extra array needed — elements are swapped in the original array.
# This is the version most commonly asked about in interviews.
# ============================================================
def partition(arr: List[int], low: int, high: int) -> int:
    """
    Partition function: partition arr[low..high].
    Returns the final index of the pivot.
    """
    # Choose the rightmost element as the pivot
    pivot = arr[high]

    # i points to the last position in the "less than pivot" region
    i = low - 1

    # Traverse from low to high-1
    for j in range(low, high):
        # If the current element <= pivot, swap it into the left region
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place the pivot in its correct position (i+1)
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1  # return the pivot index


def quicksort_inplace(arr: List[int], low: int = 0, high: int = None) -> None:
    """
    Quick sort — in-place version.
    Modifies the array directly; returns nothing.

    Args:
        arr:  array to sort
        low:  start index (default 0)
        high: end index (default len-1)
    """
    if high is None:
        high = len(arr) - 1

    # Base case
    if low < high:
        # Partition and get the pivot index
        pivot_idx = partition(arr, low, high)

        # Recursively sort left and right halves
        quicksort_inplace(arr, low, pivot_idx - 1)   # sort left
        quicksort_inplace(arr, pivot_idx + 1, high)  # sort right


# ============================================================
# Version 3: Random pivot (production-grade)
# Randomize the pivot to avoid O(n²) degradation on sorted input.
# This is the most common approach in real engineering.
# ============================================================
def partition_random(arr: List[int], low: int, high: int) -> int:
    """Partition function with a random pivot."""
    # Pick a random pivot and swap it with the last element
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]

    # The rest is the same as a regular partition
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort_random(arr: List[int], low: int = 0, high: int = None) -> None:
    """Quick sort — random pivot version."""
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_idx = partition_random(arr, low, high)
        quicksort_random(arr, low, pivot_idx - 1)
        quicksort_random(arr, pivot_idx + 1, high)


# ============================================================
# Helper: verify correctness
# ============================================================
def is_sorted(arr: List[int]) -> bool:
    """Check whether an array is sorted in ascending order."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    test_cases = [
        [3, 6, 8, 10, 1, 2, 1],
        [5, 4, 3, 2, 1],           # fully reversed
        [1, 2, 3, 4, 5],           # already sorted
        [1],                        # single element
        [],                         # empty array
        [7, 7, 7, 7, 7],           # all identical
        [-3, 10, -1, 0, 5, 2],     # contains negatives
    ]

    print("=" * 60)
    print("Quick Sort Algorithm Demo")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test case {i} ---")
        print(f"Input:    {test}")

        # Version 1: basic
        result1 = quicksort_basic(test.copy())
        print(f"Basic:    {result1}  ✅ {is_sorted(result1)}")

        # Version 2: in-place partition
        arr2 = test.copy()
        quicksort_inplace(arr2)
        print(f"In-place: {arr2}  ✅ {is_sorted(arr2)}")

        # Version 3: random pivot
        arr3 = test.copy()
        quicksort_random(arr3)
        print(f"Random:   {arr3}  ✅ {is_sorted(arr3)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
