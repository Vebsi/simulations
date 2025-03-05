# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:33:47 2024

@author: vertt
"""
#%% 
import simpy
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter



data1 = pd.read_excel("C:/Users/vertt/Desktop/etunimitilasto-2025-02-04-dvv.xlsx", sheet_name="Miehet kaikki") 
data2 = pd.read_excel("C:/Users/vertt/Desktop/etunimitilasto-2025-02-04-dvv.xlsx", sheet_name="Naiset kaikki") 
data3 = pd.read_excel("C:/Users/vertt/Desktop/sukunimitilasto-2025-02-04-dvv.xlsx",sheet_name="Nimet")

first_names_m = data1["Etunimi"]
first_names_n = data2["Etunimi"]
last_names = data3["Sukunimi"]

names = pd.concat([first_names_m, first_names_n]).reset_index(drop=True)

weight_pername_m = data1["Lukumäärä"]/sum(data1["Lukumäärä"])
weight_pername_n = data2["Lukumäärä"]/sum(data2["Lukumäärä"])

names_weight = pd.concat([weight_pername_m, weight_pername_n]).reset_index(drop=True)


last_names = data3["Sukunimi"].reset_index(drop=True)
weight_lastname = (data3["Yhteensä"] / sum(data3["Yhteensä"])).reset_index(drop=True)


# Function to generate a random population using weighted probabilities
def generate_random_population(size):
    population = [
        f"{random.choices(names, weights=names_weight)[0]} {random.choices(last_names, weights=weight_lastname)[0]}"
        for _ in range(size)
    ]
    return population


random_population = generate_random_population(150)

# Count occurrences of first and last names separately
first_names_in_population = [name.split()[0] for name in random_population]
last_names_in_population = [name.split()[1] for name in random_population]

first_name_counts = Counter(first_names_in_population)
last_name_counts = Counter(last_names_in_population)

# Get top 10 most common first and last names
top_10_first_names = first_name_counts.most_common(10)
top_10_last_names = last_name_counts.most_common(10)

# Unpack names and counts for plotting
top_10_first_names_labels, top_10_first_names_values = zip(*top_10_first_names)
top_10_last_names_labels, top_10_last_names_values = zip(*top_10_last_names)

# Plotting
plt.figure(figsize=(12, 5))

# Most common first names
plt.subplot(1, 2, 1)
plt.bar(top_10_first_names_labels, top_10_first_names_values)
plt.title('Top 10 Most Common First Names')
plt.xlabel('First Name')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # Rotating labels for readability


# Most common last names
plt.subplot(1, 2, 2)
plt.bar(top_10_last_names_labels, top_10_last_names_values)
plt.title('Top 10 Most Common Last Names')
plt.xlabel('Last Name')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # 

plt.tight_layout()
plt.show()


NUM_CLIENTS = 30000 # Customizable, higher numbers require bulkier PC
clients = generate_random_population(NUM_CLIENTS)  # Generate names for all clients

# Clients dataframe, includes different types of clients and salaries
clients_df = pd.DataFrame({
    'client_id': [f"ID{str(i).zfill(5)}" for i in range(NUM_CLIENTS)],
    'name': clients,  # Use generated names
    'service': [random.choice(['Card', 'Account', 'Both', 'Only Card', 'Only Account']) for _ in range(NUM_CLIENTS)],
    'monthly_salary': [random.randint(1500, 10000) for _ in range(NUM_CLIENTS)]
})

# Saving some memory and computing power by taking a random sample
sample_clients_df = clients_df.sample(n=10000, random_state=42)


# Opening a new bank. The best bank that will ever exist.
class Bank:
    def __init__(self, env, sample_clients_df):
        self.env = env
        self.clients = sample_clients_df.to_dict('records')  # Convert DataFrame to list of dicts
        self.transactions = []

    def salary_deposit(self, client):
        """Periodic salary deposits."""
        while True:
            yield self.env.timeout(30)  # Monthly salary deposit every 30 days
            client['balance'] = client.get('balance', 0) + client['monthly_salary']
            self.log_transaction(client['client_id'], client['monthly_salary'], 'Salary Deposit')

    # Modify the make_transaction method in the Bank class.
    def make_transaction(self, client_id, amount, transaction_type, is_sanctioned=False):
      client = next((c for c in self.clients if c['client_id'] == client_id), None)
      if client:
        allowed_transaction_types = {
            'Card': ['Retail Store', 'Online Purchase', 'Cash Deposit', 'Cash Withdrawal'],
            'Account': ['Account Transfer', 'Bill Payment'],
            'Both': ['Retail Store', 'Online Purchase', 'Account Transfer', 'Bill Payment', 'Cash Deposit', 'Cash Withdrawal'],
            'Only Card': ['Retail Store', 'Online Purchase', 'Cash Deposit', 'Cash Withdrawal'],
            'Only Account': ['Account Transfer', 'Bill Payment']
        }

        if transaction_type not in allowed_transaction_types[client['service']]:
            return  # Skip transaction if not allowed for this service type

        # Adjust balance for non-deposit/withdrawal transactions
        if transaction_type == 'Cash Deposit':
            client['balance'] = client.get('balance', 0) + abs(amount)  # Add to balance
        elif transaction_type == 'Cash Withdrawal':
            if client.get('balance', 0) >= abs(amount):  # Ensure sufficient balance
                client['balance'] -= abs(amount)
            else:
                return  # Skip if insufficient balance
        else:
            client['balance'] = client.get('balance', 0) - amount  # Normal transaction

        # Log the transaction with sanctions check
        is_fraudulent = is_sanctioned or (random.random() < 0.01)  # Flag as fraudulent if sanctioned or random chance
        unusual_behavior = is_sanctioned or self.detect_unusual_behavior(client_id, amount, transaction_type)
        self.log_transaction(client_id, amount, transaction_type, is_fraudulent, unusual_behavior)
    

    def log_transaction(self, client_id, amount, transaction_type,  is_fraudulent=False, unusual_behavior=False):
     transaction = {
        'time': self.env.now,
        'client_id': client_id,
        'amount': amount,
        'transaction_type': transaction_type,
        'is_fraudulent': is_fraudulent,
        'unusual_behavior': unusual_behavior
      }
     self.transactions.append(transaction)

    def detect_unusual_behavior(self, client_id, amount, transaction_type):
        """Identify unusual behavior based on transaction amount and type."""
        # Example conditions for unusual behavior
        if amount < -1500:  # Large negative transactions
            return True
        if random.random() < 0.005:  # Random anomaly simulation (0.5% chance)
            return True
        # More complex checks could include transaction history analysis, client behavior patterns, etc.
        return False



def client_behavior(env, bank, client):
    """Define client behavior for transactions based on their service type, including cash transactions for card users."""
    allowed_transaction_types = {
        'Card': ['Retail Store', 'Online Purchase', 'Cash Deposit', 'Cash Withdrawal'],
        'Account': ['Account Transfer', 'Bill Payment'],
        'Both': ['Retail Store', 'Online Purchase', 'Account Transfer', 'Bill Payment', 'Cash Deposit', 'Cash Withdrawal'],
        'Only Card': ['Retail Store', 'Online Purchase', 'Cash Deposit', 'Cash Withdrawal'],
        'Only Account': ['Account Transfer', 'Bill Payment']
    }
    
    countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
        "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
        "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
        "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia",
        "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica", "Côte d'Ivoire", "Croatia",
        "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor (Timor-Leste)",
        "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji",
        "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
        "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
        "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South",
        "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
        "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands",
        "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco",
        "Mozambique", "Myanmar (Burma)", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger",
        "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru",
        "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia",
        "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa",
        "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan",
        "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
        "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
        "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
    ]

    highrisk_countries = ["Afghanistan", "Angola","Armenia","Azerbaijan", "Belarus","Bhutan", "Bulgaria", "Cambodia", "Cameroon",
                          "Central African Republic", "Chad","China","Congo, Democratic Republic of the", "Congo, Republic of the",
                          "Côte d'Ivoire","Ethiopia","Gabon", "Gambia", "Ghana","India","Iran", "Iraq","Israel","Jordan", "Kazakhstan",
                          "Kenya", "Kiribati", "Korea, North","Kuwait", "Kyrgyzstan", "Laos","Libya","Montenegro", "Morocco","Myanmar (Burma)",
                          "Namibia", "Nauru","Nigeria","Pakistan", "Somalia", "South Africa","Niger","Qatar","Russia","Tanzania", "Thailand",
                          "Tunisia", "Turkey", "Turkmenistan","Ukraine", "United Arab Emirates","Uzbekistan","Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"]

    # Define weights for transaction types (higher values = more common)
    transaction_weights = {
        'Retail Store': 15,
        'Online Purchase': 9,
        'Account Transfer': 10,
        'Bill Payment': 5,
        'Cash Deposit': 6,
        'Cash Withdrawal': 4
    }

    while True:
        yield env.timeout(random.randint(1, 7))  # Random delay between transactions
        
        # Get allowed transactions and their weights for the client
        client_allowed_types = allowed_transaction_types[client['service']]
        client_weights = [transaction_weights[tran_type] for tran_type in client_allowed_types]

        # Select a transaction type using weights
        transaction_type = random.choices(client_allowed_types, weights=client_weights, k=1)[0]
        
        # Generate a random amount
        amount = round(random.uniform(5, 5000), 2)  # Random transaction amount
        
        # Handle special case for Account Transfers
        if transaction_type == 'Account Transfer':
            transfer_type = random.choice(['Domestic', 'Foreign'])
            if transfer_type == 'Foreign':
                destination_country = random.choice(countries)
                transaction_type = f'Foreign Transfer to {destination_country}'
                if destination_country in highrisk_countries:
                    bank.make_transaction(client['client_id'], -amount, transaction_type, is_sanctioned=True)
                    continue  # Flagged as fraudulent and skip further processing
                else:
                     transaction_type = 'Domestic Transfer'
       # Handle cash-specific logic
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal']:
          if client['service'] not in ['Card', 'Both', 'Only Card']:
             continue  # Skip if client does not have a card
            
            # Special handling for cash deposits/withdrawals
          if transaction_type == 'Cash Deposit':
                bank.make_transaction(client['client_id'], amount, transaction_type)  # Add amount
          elif transaction_type == 'Cash Withdrawal':
                bank.make_transaction(client['client_id'], -amount, transaction_type)  # Subtract amount
        else:
            # Handle other transaction types normally
            bank.make_transaction(client['client_id'], amount, transaction_type)


# Simulating the environment
env = simpy.Environment()
bank = Bank(env, sample_clients_df)

# Creating processes for each client
for client in bank.clients:
    env.process(bank.salary_deposit(client))  # Schedule salary deposits
    env.process(client_behavior(env, bank, client))  # Schedule client behaviors

# Run the simulation for 365 days
env.run(until=365)

# Convert transactions to DataFrame for analysis
transactions_df = pd.DataFrame(bank.transactions)


# Saving imaginary data to my FantaSQL server (commented out)

#from sqlalchemy import create_engine

# Connection string
#engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server')

# Save DataFrame to SQL Server
#df.to_sql('transactions', con=engine, if_exists='append', index=False)


#Hypothetically fetching data from above

#DATABASE_URL = "mssql+pyodbc://username:password@server/database?driver=ABCD+Driver+17+for+SQL+Server"
#engine = create_engine(DATABASE_URL)

# As the whole data file is wanted, everything is selected

#query = "SELECT * FROM transactions"
#df = pd.read_sql(query, con=engine)
#print(df.head())


# Creating a function that fetches unusual behaviour from dataset with customizable date range

def plot_unusual_transactions_custom_range(transactions_df, start_time, end_time):
    # Filter unusual transactions within the custom time range
    unusual_transactions = transactions_df[
        (transactions_df['unusual_behavior'] == True) & 
        (transactions_df['time'] >= start_time) & 
        (transactions_df['time'] <= end_time)
    ].reset_index()
    # Group by date and count occurrences
    unusual_per_day = unusual_transactions.groupby("time")["amount"].count().reset_index()
    unusual_per_day.columns = ["time", "count"]  # Rename columns for clarity
    # Plot the unusual transactions per day
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=unusual_per_day, x="time", y="count")
    
    plt.xlabel("Date")
    plt.ylabel("Unusual Transactions Count")
    plt.title(f"Unusual Transactions Per Day (Time {start_time} to {end_time})")
    
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    return unusual_per_day  # Return the DataFrame for further analysis

unusual_data = plot_unusual_transactions_custom_range(transactions_df, 10, 20)

unusual_data1 = plot_unusual_transactions_custom_range(transactions_df, 17, 150)
