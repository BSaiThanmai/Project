import random

def number_guessing_game():
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100. Can you guess what it is?")
    
    # Generate a random number between 1 and 100
    target_number = random.randint(1, 100)
    attempts = 0  # Counter for attempts

    while True:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1

            if guess < target_number:
                print("Too low! Try again.")
            elif guess > target_number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You've guessed it in {attempts} attempts.")
                break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
    
    print("Thanks for playing!")

# Run the game
if __name__ == "__main__":
    number_guessing_game()

