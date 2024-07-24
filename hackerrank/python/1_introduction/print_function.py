def main():
    n = int(input())

    if n in range(1,151): 
        res = ''

        for i in range(1,n+1):
            res = res + str(i)

        print(res)

    else:
        print("Please enter a value between 1 and 150.")

if __name__ == '__main__':
    main()