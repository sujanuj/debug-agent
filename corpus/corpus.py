"""Bug corpus for the autonomous debugging agent.

Each entry is a BugCase with:
  - id: unique identifier
  - language: python | javascript | go
  - description: what the function is supposed to do
  - buggy_code: the broken implementation
  - test_code: tests that fail against the buggy code
  - fixed_code: the correct implementation (used to verify agent's fix)
  - bug_type: category of bug (off-by-one, wrong-operator, missing-case, etc.)

The corpus covers 3 languages and 6 bug types, with 5 bugs per language
(15 total). All bugs are realistic -- the kind that appear in real code
reviews and interview problems, not contrived syntax errors.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BugCase:
    id: str
    language: str           # python | javascript | go
    description: str
    buggy_code: str
    test_code: str
    fixed_code: str
    bug_type: str           # off-by-one | wrong-operator | missing-case |
                            # wrong-return | logic-error | boundary

    def __repr__(self):
        return f"BugCase(id={self.id!r}, lang={self.language!r}, type={self.bug_type!r})"


# ---------------------------------------------------------------------------
# Python bugs
# ---------------------------------------------------------------------------

PYTHON_BUGS = [
    BugCase(
        id="py-001",
        language="python",
        bug_type="off-by-one",
        description="Return the second largest element in a list.",
        buggy_code='''\
def second_largest(nums):
    if len(nums) < 2:
        return None
    sorted_nums = sorted(set(nums))
    return sorted_nums[-1]  # BUG: returns largest, not second largest
''',
        test_code='''\
from solution import second_largest

def test_basic():
    assert second_largest([3, 1, 4, 1, 5, 9, 2, 6]) == 6

def test_two_elements():
    assert second_largest([1, 2]) == 1

def test_with_duplicates():
    assert second_largest([5, 5, 3]) == 3

def test_returns_none_for_single():
    assert second_largest([42]) is None

if __name__ == "__main__":
    test_basic()
    test_two_elements()
    test_with_duplicates()
    test_returns_none_for_single()
    print("ALL TESTS PASSED")
''',
        fixed_code='''\
def second_largest(nums):
    if len(nums) < 2:
        return None
    sorted_nums = sorted(set(nums))
    if len(sorted_nums) < 2:
        return None
    return sorted_nums[-2]
''',
    ),

    BugCase(
        id="py-002",
        language="python",
        bug_type="wrong-operator",
        description="Check if a number is prime.",
        buggy_code='''\
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return False  # BUG: 2 is prime, this incorrectly returns False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
        test_code='''\
from solution import is_prime

def test_small_primes():
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(5) == True
    assert is_prime(7) == True

def test_non_primes():
    assert is_prime(1) == False
    assert is_prime(4) == False
    assert is_prime(9) == False

def test_large_prime():
    assert is_prime(97) == True

def test_large_non_prime():
    assert is_prime(100) == False

if __name__ == "__main__":
    test_small_primes()
    test_non_primes()
    test_large_prime()
    test_large_non_prime()
    print("ALL TESTS PASSED")
''',
        fixed_code='''\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
    ),

    BugCase(
        id="py-003",
        language="python",
        bug_type="missing-case",
        description="Flatten a nested list one level deep.",
        buggy_code='''\
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)
        # BUG: missing else branch -- non-list items are dropped
    return result
''',
        test_code='''\
from solution import flatten

def test_basic():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

def test_mixed():
    assert flatten([[1, 2], 3, [4, 5]]) == [1, 2, 3, 4, 5]

def test_empty():
    assert flatten([]) == []

def test_no_nesting():
    assert flatten([1, 2, 3]) == [1, 2, 3]

if __name__ == "__main__":
    test_basic()
    test_mixed()
    test_empty()
    test_no_nesting()
    print("ALL TESTS PASSED")
''',
        fixed_code='''\
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result
''',
    ),

    BugCase(
        id="py-004",
        language="python",
        bug_type="logic-error",
        description="Count the number of vowels in a string.",
        buggy_code='''\
def count_vowels(s):
    vowels = "aeiou"
    count = 0
    for char in s:
        if char in vowels:  # BUG: misses uppercase vowels
            count += 1
    return count
''',
        test_code='''\
from solution import count_vowels

def test_lowercase():
    assert count_vowels("hello") == 2

def test_uppercase():
    assert count_vowels("HELLO") == 2

def test_mixed():
    assert count_vowels("Hello World") == 3

def test_empty():
    assert count_vowels("") == 0

def test_no_vowels():
    assert count_vowels("rhythm") == 0

if __name__ == "__main__":
    test_lowercase()
    test_uppercase()
    test_mixed()
    test_empty()
    test_no_vowels()
    print("ALL TESTS PASSED")
''',
        fixed_code='''\
def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count
''',
    ),

    BugCase(
        id="py-005",
        language="python",
        bug_type="boundary",
        description="Binary search: return index of target in sorted list, or -1.",
        buggy_code='''\
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:  # BUG: should be left <= right
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
        test_code='''\
from solution import binary_search

def test_found_middle():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2

def test_found_first():
    assert binary_search([1, 3, 5, 7, 9], 1) == 0

def test_found_last():
    assert binary_search([1, 3, 5, 7, 9], 9) == 4

def test_not_found():
    assert binary_search([1, 3, 5, 7, 9], 4) == -1

def test_single_element_found():
    assert binary_search([42], 42) == 0

def test_single_element_not_found():
    assert binary_search([42], 1) == -1

if __name__ == "__main__":
    test_found_middle()
    test_found_first()
    test_found_last()
    test_not_found()
    test_single_element_found()
    test_single_element_not_found()
    print("ALL TESTS PASSED")
''',
        fixed_code='''\
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
    ),
]

# ---------------------------------------------------------------------------
# JavaScript bugs
# ---------------------------------------------------------------------------

JAVASCRIPT_BUGS = [
    BugCase(
        id="js-001",
        language="javascript",
        bug_type="wrong-operator",
        description="Check if two strings are anagrams.",
        buggy_code='''\
function areAnagrams(s1, s2) {
    if (s1.length !== s2.length) return false;
    const sorted1 = s1.split("").sort().join("");
    const sorted2 = s2.split("").sort().join("");
    return sorted1 == sorted2; // BUG: should use === for strict equality
}
module.exports = { areAnagrams };
''',
        test_code='''\
const { areAnagrams } = require("./solution");

function assert(condition, message) {
    if (!condition) throw new Error(message || "Assertion failed");
}

assert(areAnagrams("listen", "silent") === true, "listen/silent");
assert(areAnagrams("hello", "world") === false, "hello/world");
assert(areAnagrams("anagram", "nagaram") === true, "anagram/nagaram");
assert(areAnagrams("rat", "car") === false, "rat/car");
assert(areAnagrams("", "") === true, "empty strings");

console.log("ALL TESTS PASSED");
''',
        fixed_code='''\
function areAnagrams(s1, s2) {
    if (s1.length !== s2.length) return false;
    const sorted1 = s1.split("").sort().join("");
    const sorted2 = s2.split("").sort().join("");
    return sorted1 === sorted2;
}
module.exports = { areAnagrams };
''',
    ),

    BugCase(
        id="js-002",
        language="javascript",
        bug_type="off-by-one",
        description="Reverse a string.",
        buggy_code='''\
function reverseString(s) {
    let result = "";
    for (let i = s.length - 1; i > 0; i--) { // BUG: i > 0 skips first char
        result += s[i];
    }
    return result;
}
module.exports = { reverseString };
''',
        test_code='''\
const { reverseString } = require("./solution");

function assert(condition, message) {
    if (!condition) throw new Error(message || "Assertion failed");
}

assert(reverseString("hello") === "olleh", "hello");
assert(reverseString("a") === "a", "single char");
assert(reverseString("") === "", "empty");
assert(reverseString("abcd") === "dcba", "abcd");

console.log("ALL TESTS PASSED");
''',
        fixed_code='''\
function reverseString(s) {
    let result = "";
    for (let i = s.length - 1; i >= 0; i--) {
        result += s[i];
    }
    return result;
}
module.exports = { reverseString };
''',
    ),

    BugCase(
        id="js-003",
        language="javascript",
        bug_type="logic-error",
        description="Find the maximum value in an array.",
        buggy_code='''\
function findMax(arr) {
    if (arr.length === 0) return null;
    let max = 0; // BUG: should initialize to arr[0], not 0
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}
module.exports = { findMax };
''',
        test_code='''\
const { findMax } = require("./solution");

function assert(condition, message) {
    if (!condition) throw new Error(message || "Assertion failed");
}

assert(findMax([3, 1, 4, 1, 5, 9]) === 9, "positive numbers");
assert(findMax([-3, -1, -4]) === -1, "all negative");
assert(findMax([0]) === 0, "single zero");
assert(findMax([]) === null, "empty array");

console.log("ALL TESTS PASSED");
''',
        fixed_code='''\
function findMax(arr) {
    if (arr.length === 0) return null;
    let max = arr[0];
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}
module.exports = { findMax };
''',
    ),

    BugCase(
        id="js-004",
        language="javascript",
        bug_type="missing-case",
        description="Capitalize the first letter of each word in a string.",
        buggy_code='''\
function titleCase(str) {
    return str.split(" ").map(word => {
        return word[0].toUpperCase() + word.slice(1); // BUG: crashes on empty string
    }).join(" ");
}
module.exports = { titleCase };
''',
        test_code='''\
const { titleCase } = require("./solution");

function assert(condition, message) {
    if (!condition) throw new Error(message || "Assertion failed");
}

assert(titleCase("hello world") === "Hello World", "basic");
assert(titleCase("the quick brown fox") === "The Quick Brown Fox", "sentence");
assert(titleCase("") === "", "empty string");
assert(titleCase("a") === "A", "single char");

console.log("ALL TESTS PASSED");
''',
        fixed_code='''\
function titleCase(str) {
    if (str === "") return "";
    return str.split(" ").map(word => {
        if (word.length === 0) return word;
        return word[0].toUpperCase() + word.slice(1);
    }).join(" ");
}
module.exports = { titleCase };
''',
    ),

    BugCase(
        id="js-005",
        language="javascript",
        bug_type="wrong-return",
        description="Sum all numbers in a nested array (one level deep).",
        buggy_code='''\
function sumNested(arr) {
    let total = 0;
    for (const item of arr) {
        if (Array.isArray(item)) {
            item.reduce((acc, val) => acc + val, 0); // BUG: result not added to total
        } else {
            total += item;
        }
    }
    return total;
}
module.exports = { sumNested };
''',
        test_code='''\
const { sumNested } = require("./solution");

function assert(condition, message) {
    if (!condition) throw new Error(message || "Assertion failed");
}

assert(sumNested([1, [2, 3], 4]) === 10, "basic");
assert(sumNested([[1, 2], [3, 4]]) === 10, "all nested");
assert(sumNested([1, 2, 3]) === 6, "flat");
assert(sumNested([]) === 0, "empty");

console.log("ALL TESTS PASSED");
''',
        fixed_code='''\
function sumNested(arr) {
    let total = 0;
    for (const item of arr) {
        if (Array.isArray(item)) {
            total += item.reduce((acc, val) => acc + val, 0);
        } else {
            total += item;
        }
    }
    return total;
}
module.exports = { sumNested };
''',
    ),
]

# ---------------------------------------------------------------------------
# Go bugs
# ---------------------------------------------------------------------------

GO_BUGS = [
    BugCase(
        id="go-001",
        language="go",
        bug_type="off-by-one",
        description="Return the nth Fibonacci number (0-indexed).",
        buggy_code='''\
package solution

func Fibonacci(n int) int {
    if n <= 0 {
        return 0
    }
    if n == 1 {
        return 1
    }
    a, b := 0, 1
    for i := 2; i < n; i++ { // BUG: should be i <= n
        a, b = b, a+b
    }
    return b
}
''',
        test_code='''\
package solution

import "testing"

func TestFibonacci(t *testing.T) {
    cases := []struct {
        n, want int
    }{
        {0, 0}, {1, 1}, {2, 1}, {3, 2},
        {4, 3}, {5, 5}, {6, 8}, {10, 55},
    }
    for _, c := range cases {
        got := Fibonacci(c.n)
        if got != c.want {
            t.Errorf("Fibonacci(%d) = %d, want %d", c.n, got, c.want)
        }
    }
}
''',
        fixed_code='''\
package solution

func Fibonacci(n int) int {
    if n <= 0 {
        return 0
    }
    if n == 1 {
        return 1
    }
    a, b := 0, 1
    for i := 2; i <= n; i++ {
        a, b = b, a+b
    }
    return b
}
''',
    ),

    BugCase(
        id="go-002",
        language="go",
        bug_type="logic-error",
        description="Check if a string is a palindrome.",
        buggy_code='''\
package solution

func IsPalindrome(s string) bool {
    n := len(s)
    for i := 0; i < n/2; i++ {
        if s[i] != s[n-i] { // BUG: should be s[n-1-i]
            return false
        }
    }
    return true
}
''',
        test_code='''\
package solution

import "testing"

func TestIsPalindrome(t *testing.T) {
    cases := []struct {
        s    string
        want bool
    }{
        {"racecar", true},
        {"hello", false},
        {"a", true},
        {"", true},
        {"abba", true},
        {"abcd", false},
    }
    for _, c := range cases {
        got := IsPalindrome(c.s)
        if got != c.want {
            t.Errorf("IsPalindrome(%q) = %v, want %v", c.s, got, c.want)
        }
    }
}
''',
        fixed_code='''\
package solution

func IsPalindrome(s string) bool {
    n := len(s)
    for i := 0; i < n/2; i++ {
        if s[i] != s[n-1-i] {
            return false
        }
    }
    return true
}
''',
    ),

    BugCase(
        id="go-003",
        language="go",
        bug_type="wrong-operator",
        description="Count occurrences of each character in a string.",
        buggy_code='''\
package solution

func CharCount(s string) map[rune]int {
    counts := make(map[rune]int)
    for _, ch := range s {
        counts[ch] = 1 // BUG: should be counts[ch]++
    }
    return counts
}
''',
        test_code='''\
package solution

import "testing"

func TestCharCount(t *testing.T) {
    got := CharCount("hello")
    expected := map[rune]int{
        'h': 1, 'e': 1, 'l': 2, 'o': 1,
    }
    for ch, want := range expected {
        if got[ch] != want {
            t.Errorf("CharCount(%q)[%c] = %d, want %d", "hello", ch, got[ch], want)
        }
    }

    got2 := CharCount("")
    if len(got2) != 0 {
        t.Errorf("CharCount(\"\") should be empty map")
    }
}
''',
        fixed_code='''\
package solution

func CharCount(s string) map[rune]int {
    counts := make(map[rune]int)
    for _, ch := range s {
        counts[ch]++
    }
    return counts
}
''',
    ),

    BugCase(
        id="go-004",
        language="go",
        bug_type="missing-case",
        description="Reverse a slice of integers in place.",
        buggy_code='''\
package solution

func ReverseSlice(nums []int) {
    n := len(nums)
    for i := 0; i < n/2; i++ {
        // BUG: swaps wrong indices
        nums[i], nums[n-i] = nums[n-i], nums[i]
    }
}
''',
        test_code='''\
package solution

import (
    "reflect"
    "testing"
)

func TestReverseSlice(t *testing.T) {
    cases := []struct {
        input, want []int
    }{
        {[]int{1, 2, 3, 4, 5}, []int{5, 4, 3, 2, 1}},
        {[]int{1, 2}, []int{2, 1}},
        {[]int{1}, []int{1}},
        {[]int{}, []int{}},
    }
    for _, c := range cases {
        ReverseSlice(c.input)
        if !reflect.DeepEqual(c.input, c.want) {
            t.Errorf("got %v, want %v", c.input, c.want)
        }
    }
}
''',
        fixed_code='''\
package solution

func ReverseSlice(nums []int) {
    n := len(nums)
    for i := 0; i < n/2; i++ {
        nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
    }
}
''',
    ),

    BugCase(
        id="go-005",
        language="go",
        bug_type="boundary",
        description="Find the first duplicate in a slice, or -1 if none.",
        buggy_code='''\
package solution

func FirstDuplicate(nums []int) int {
    seen := make(map[int]bool)
    for _, n := range nums {
        if seen[n] {
            return n
        }
        seen[n] = false // BUG: should be true
    }
    return -1
}
''',
        test_code='''\
package solution

import "testing"

func TestFirstDuplicate(t *testing.T) {
    cases := []struct {
        nums []int
        want int
    }{
        {[]int{1, 2, 3, 2, 1}, 2},
        {[]int{1, 2, 3, 4}, -1},
        {[]int{5, 5}, 5},
        {[]int{}, -1},
    }
    for _, c := range cases {
        got := FirstDuplicate(c.nums)
        if got != c.want {
            t.Errorf("FirstDuplicate(%v) = %d, want %d", c.nums, got, c.want)
        }
    }
}
''',
        fixed_code='''\
package solution

func FirstDuplicate(nums []int) int {
    seen := make(map[int]bool)
    for _, n := range nums {
        if seen[n] {
            return n
        }
        seen[n] = true
    }
    return -1
}
''',
    ),
]

# ---------------------------------------------------------------------------
# Combined corpus
# ---------------------------------------------------------------------------

ALL_BUGS: List[BugCase] = PYTHON_BUGS + JAVASCRIPT_BUGS + GO_BUGS


def get_bug(bug_id: str) -> Optional[BugCase]:
    for bug in ALL_BUGS:
        if bug.id == bug_id:
            return bug
    return None


def get_bugs_by_language(language: str) -> List[BugCase]:
    return [b for b in ALL_BUGS if b.language == language]


def get_bugs_by_type(bug_type: str) -> List[BugCase]:
    return [b for b in ALL_BUGS if b.bug_type == bug_type]


if __name__ == "__main__":
    print(f"Total bugs: {len(ALL_BUGS)}")
    for lang in ["python", "javascript", "go"]:
        bugs = get_bugs_by_language(lang)
        print(f"  {lang}: {len(bugs)} bugs")
    print()
    types = set(b.bug_type for b in ALL_BUGS)
    for t in sorted(types):
        bugs = get_bugs_by_type(t)
        print(f"  {t}: {len(bugs)} bugs")
