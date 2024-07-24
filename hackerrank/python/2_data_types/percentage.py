def main():
    n = int(input())

    student_marks = {}

    names = []

    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores

    query_name = input()



if __name__ == '__main__':
    main()