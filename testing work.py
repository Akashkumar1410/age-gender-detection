import pandas as pd

class CustomerData():
    customer_data = dict()

    def save_data(self, customer_account, customer_name, balance):
        customer_account = customer_account.strip()
        customer_name = customer_name.strip()
        if customer_account in self.customer_data:
            # Customer exists, update the balance
            self.customer_data[customer_account]['balance'] += balance
        else:
            # Customer doesn't exist, initialize the balance
            self.customer_data[customer_account] = {'name': customer_name, 'balance': balance}
        #print(f'Data saved for Account Number {customer_account} - Balance: {balance}.')

    def display_all_details(self):
        print("\nAll Customer Details:")
        for account, details in self.customer_data.items():
            print(f'Account Number: {account} - Customer: {details["name"]} - Balance: {details["balance"]}')

        # Convert customer_data to a DataFrame
        df = pd.DataFrame(list(self.customer_data.items()), columns=['Account', 'Details'])

        # Save DataFrame to Excel
        df.to_excel('customer_data.xlsx', index=False)
        print("\nCustomer data saved to 'customer_data.xlsx'.")

class Bank(CustomerData):
    def __init__(self, aaccount, aowner, abalance=0):
        super().__init__()
        self.account = aaccount
        self.owner = aowner
        self.balance = abalance

    def deposit(self, amount):
        self.balance += amount
        print(f'Your amount {amount} has been deposited to your account.')
        print(f'Your total balance is {self.balance}')
        # Save data after each transaction
        self.save_data(self.account, self.owner, self.balance)

    def withdraw(self, credit):
        if self.balance >= credit:
            self.balance -= credit
            print(f'You withdrew {credit} and the remaining balance is {self.balance}')
            # Save data after each transaction
            self.save_data(self.account, self.owner, -credit)  # Update balance with a negative value for withdrawal
        else:
            print("Insufficient balance.")

    def details(self):
        print(f'Account Number: {self.account} - Name of the customer: {self.owner} Balance: {self.balance}')
        # Save data after each transaction
        self.save_data(self.account, self.owner, self.balance)

# Creating an instance of CustomerData to manage customer details
customer_manager = CustomerData()

print("Enter 'n' for new customer account\nPress 'd' for Deposit\nPress 'w' for Withdraw\nPress 'e' for Details\nEnter 'q' for Exit press any key for details")

account1 = None

while True:
    choice = input("*************Enter your choice***********")
    if choice == 'n':
        print("Enter the customer account number")
        account_number = input("Enter account number: ")
        print("Enter the customer name")
        customer_name = input("Enter name: ")
        if account_number.strip() not in CustomerData.customer_data:
            CustomerData.customer_data[account_number.strip()] = {'name': customer_name, 'balance': 0}  # Initialize balance to 0
            account1 = Bank(account_number, customer_name)  # Create Bank instance only if it doesn't exist
        else:
            account1 = Bank(account_number, customer_name, CustomerData.customer_data[account_number.strip()]['balance'])  # Use existing balance

    elif choice == 'd':
        amount_for_deposit = int(input("Enter the amount you want to deposit: "))
        account1.deposit(amount_for_deposit)

    elif choice == 'w':
        if account1:
            print("Enter the amount you want to withdraw")
            amount_for_withdraw = int(input("Enter your amount: "))
            account1.withdraw(amount_for_withdraw)
        else:
            print("Create an account first (press 'n')")

    elif choice == 'e':
        if account1:
            account1.details()
        else:
            print("Create an account first (press 'n')")

    elif choice == 'q':
        print("Thank you for visiting.")
        break

# Continuous display of all customer names and details
for account, details in CustomerData.customer_data.items():
    print(f'Account Number: {account} - Customer: {details["name"]} - Balance: {details["balance"]}')

# Display all customer details at the end
customer_manager.display_all_details()
