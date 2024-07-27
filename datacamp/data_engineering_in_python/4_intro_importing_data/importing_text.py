# Open a file: file
file = open('datasets/mobydick.txt', 'r')

def read_entire_file(file):
    # Print it
    print(file.read())

    # Check whether file is closed
    print(file.closed)

    # Close file
    file.close()

    # Check whether file is closed
    print(file.closed)

def read_line_by_line(file):
    with file as file:
        print(file.readline())
        print(file.readline())
        print(file.readline())

read_line_by_line(file)