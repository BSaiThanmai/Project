import math

class Calculator:
    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

    def multiply(self, x, y):
        return x * y

    def divide(self, x, y):
        if y == 0:
            return "Error: Division by zero"
        return x / y

    def exponentiate(self, x, y):
        return x ** y

    def square_root(self, x):
        if x < 0:
            return "Error: Cannot take square root of a negative number"
        return math.sqrt(x)

def some_function():
    calc = Calculator()

    print("Welcome to the Advanced Calculator!")
    
    while True:
        print("\nSelect operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exponentiation")
        print("6. Square Root")
        print("q. Quit")
        
        choice = input("Enter choice: ")
        
        if choice == 'q':
            print("Exiting the calculator. Goodbye!")
            break
        
        if choice in ['1', '2', '3', '4', '5', '6']:
            try:
                if choice == '6':
                    num = float(input("Enter number: "))
                    result = calc.square_root(num)
                else:
                    num1 = float(input("Enter first number: "))
                    num2 = float(input("Enter second number: "))
                    if choice == '1':
                        result = calc.add(num1, num2)
                    elif choice == '2':
                        result = calc.subtract(num1, num2)
                    elif choice == '3':
                        result = calc.multiply(num1, num2)
                    elif choice == '4':
                        result = calc.divide(num1, num2)
                    elif choice == '5':
                        result = calc.exponentiate(num1, num2)
                
                print(f"Result: {result}")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        else:
            print("Invalid choice. Please select a valid operation.")

if __name__ == "__main__":
    some_function()
