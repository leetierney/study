def main():
    students = []

    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])

    # Extract unique scores and sort them
    unique_scores = sorted(set(student[1] for student in students))

    # Find the second lowest score
    if len(unique_scores) > 1:
        second_lowest_score = unique_scores[1]
    else:
        print("Not enough unique scores to determine the second lowest score.")
        return

    # Find all students with the second lowest score
    second_lowest_students = [student[0] for student in students if student[1] == second_lowest_score]

    # Print names of students with the second lowest score, sorted alphabetically
    for name in sorted(second_lowest_students):
        print(name)

if __name__ == '__main__':
    main()
