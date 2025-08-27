import sqlite3

# Connect to SQLite database (creates file if it doesn't exist)
connection = sqlite3.connect("company_employees.db")

# Create a cursor object for executing SQL commands
cursor = connection.cursor()

# Create the STUDENT table with updated column names and appropriate types
table_info = """
CREATE TABLE IF NOT EXISTS EMPLOYEE(
    NAME VARCHAR(50),
    DEPARTMENT VARCHAR(50),
    TEAM VARCHAR(50),
    PERFORMANCE_SCORE INT
)
"""
cursor.execute(table_info)

# Insert a larger dataset of employee records representing various departments and teams
employees = [
    ('Alice Johnson', 'Engineering', 'Backend', 92),
    ('Bob Smith', 'Engineering', 'Frontend', 85),
    ('Carol Martinez', 'Engineering', 'DevOps', 88),
    ('David Lee', 'Sales', 'Domestic', 79),
    ('Eva Walker', 'Sales', 'International', 83),
    ('Frank Turner', 'Marketing', 'Content', 90),
    ('Grace Young', 'Marketing', 'SEO', 87),
    ('Hank Scott', 'HR', 'Recruitment', 78),
    ('Ivy Adams', 'HR', 'Employee Relations', 82),
    ('Jack Brown', 'Finance', 'Accounts Payable', 76),
    ('Kara Wilson', 'Finance', 'Accounts Receivable', 80),
    ('Liam King', 'Engineering', 'Backend', 91),
    ('Mia Rodriguez', 'Engineering', 'Frontend', 89),
    ('Nina Patel', 'Engineering', 'DevOps', 85),
    ('Oscar Clark', 'Sales', 'Domestic', 83),
    ('Pamela Lewis', 'Sales', 'International', 81),
    ('Quinn Hall', 'Marketing', 'Content', 79),
    ('Rachel Allen', 'Marketing', 'SEO', 88),
    ('Sam Green', 'HR', 'Recruitment', 75),
    ('Tina Nelson', 'HR', 'Employee Relations', 84),
    ('Uma Reed', 'Finance', 'Accounts Payable', 77),
    ('Victor Young', 'Finance', 'Accounts Receivable', 82),
    ('Wendy Carter', 'Engineering', 'Backend', 90),
    ('Xander Mitchell', 'Engineering', 'Frontend', 86),
    ('Yara Perez', 'Engineering', 'DevOps', 87),
    ('Zachary Collins', 'Sales', 'Domestic', 80),
    ('Ava Brooks', 'Sales', 'International', 85),
    ('Benjamin Foster', 'Marketing', 'Content', 83),
    ('Chloe Diaz', 'Marketing', 'SEO', 89),
    ('Derek Hayes', 'HR', 'Recruitment', 78),
    ('Ella Morgan', 'HR', 'Employee Relations', 81),
    ('Felix James', 'Finance', 'Accounts Payable', 79),
    ('Gabriella Sanchez', 'Finance', 'Accounts Receivable', 84)
]

# Insert all employee data into the database
cursor.executemany("INSERT INTO EMPLOYEE VALUES (?, ?, ?, ?)", employees)

# Commit changes
connection.commit()

# Display the records
print("The inserted employee records are:")
data = cursor.execute("SELECT * FROM EMPLOYEE")

for row in data:
    print(row)

# Close database connection
connection.close()
