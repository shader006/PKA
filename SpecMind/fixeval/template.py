template_code = r"""
from io import StringIO
import sys

class PostconditionCatchBug(Exception):
    pass

def run(__src: str) -> str:
    lines = __src.splitlines()
    __it = iter(lines)
    __out = []
    sys.stdin = StringIO(__src)

    def print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        __out.append(sep.join(
            f"{{a:0f}}" if isinstance(a, float) else str(a)
            for a in args
        ) + end)

    def main():
        # --- original code (unmodified logic) ---
{code}
        # --- end original code ---

    main()
    return_values: list[str] = "".join(__out).splitlines()

    try:
    # --- begin postcondition ---
{postcondition}
    # --- end postcondition ---
    except AssertionError as e:
        raise PostconditionCatchBug(e)

    return return_values

if __name__ == "__main__":
    import sys
    print("\n".join(run(sys.stdin.read())))
"""

def transform_code_with_postcondition(code, postcondition):
    indent = "    "
    indent2 = indent * 2
    formatted_code = indent2 + code.replace("\n", "\n" + indent2)
    formatted_postcondition = indent2 + postcondition.replace("\n", "\n" + indent2)
    return template_code.format(code=formatted_code, postcondition=formatted_postcondition)


if __name__ == "__main__":
    code = """
N = int(input())
z, w = [], []
K = 0
for i in range(N):
    x, y = map(int, input().split())
    z.append(x)
    w.append(y)
if N >= 3:
    for j in range(N - 2):
        if z[j] == w[j] and z[j + 1] == w[j + 1] and z[j + 2] == w[j + 2]:
            K += 1
if K >= 1:
    print("Yes")
else:
    print("No")
    """
    postcondition = """
assert return_values[0] == "Yes"
    """
    print(transform_code_with_postcondition(code, postcondition))