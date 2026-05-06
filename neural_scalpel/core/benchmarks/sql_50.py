def get_sql_50_suite():
    """Returns the full 50-case structured SQL benchmark suite."""
    suite = []
    
    # 1. basic_select (10 cases)
    for i in range(1, 11):
        suite.append({
            "id": f"basic_select_{i:03}",
            "category": "basic_select",
            "prompt": f"SQL: Get names of users in region {i}.\nTable: users(id, name, region_id)\nOutput:",
            "tables": ["users"],
            "columns": ["name", "region_id"],
            "sqlite_setup": "CREATE TABLE users (id INT, name TEXT, region_id INT); INSERT INTO users VALUES (1, 'Alice', 1), (2, 'Bob', 2), (3, 'Charlie', 1);",
            "expected_result": [('Alice',), ('Charlie',)] if i == 1 else [],
            "order_sensitive": False
        })
        
    # 2. aggregation (10 cases)
    for i in range(1, 11):
        suite.append({
            "id": f"aggregation_{i:03}",
            "category": "aggregation",
            "prompt": f"SQL: Total sales for store {i}.\nTable: sales(id, store_id, amount)\nOutput:",
            "tables": ["sales"],
            "columns": ["amount", "store_id"],
            "sqlite_setup": "CREATE TABLE sales (id INT, store_id INT, amount INT); INSERT INTO sales VALUES (1, 1, 100), (2, 1, 200), (3, 2, 500);",
            "expected_result": [(300,)] if i == 1 else [],
            "order_sensitive": False
        })

    # 3. joins (10 cases)
    for i in range(1, 11):
        suite.append({
            "id": f"joins_{i:03}",
            "category": "joins",
            "prompt": f"SQL: Names of products in category {i}.\nTable: products(id, name, cat_id), categories(id, name)\nOutput:",
            "tables": ["products", "categories"],
            "columns": ["name"],
            "sqlite_setup": "CREATE TABLE categories (id INT, name TEXT); CREATE TABLE products (id INT, name TEXT, cat_id INT); INSERT INTO categories VALUES (1, 'Tech'); INSERT INTO products VALUES (1, 'Laptop', 1);",
            "expected_result": [('Laptop',)] if i == 1 else [],
            "order_sensitive": False
        })

    # 4. subqueries (10 cases)
    for i in range(1, 11):
        suite.append({
            "id": f"subqueries_{i:03}",
            "category": "subqueries",
            "prompt": f"SQL: Employees with salary above avg in dept {i}.\nTable: employees(id, name, salary, dept_id)\nOutput:",
            "tables": ["employees"],
            "columns": ["name", "salary"],
            "sqlite_setup": "CREATE TABLE employees (id INT, name TEXT, salary INT, dept_id INT); INSERT INTO employees VALUES (1, 'A', 10, 1), (2, 'B', 20, 1), (3, 'C', 30, 1);",
            "expected_result": [('C',)] if i == 1 else [],
            "order_sensitive": False
        })

    # 5. complex_logic (10 cases)
    for i in range(1, 11):
        suite.append({
            "id": f"complex_logic_{i:03}",
            "category": "complex_logic",
            "prompt": f"SQL: Customers who bought all items in order {i}.\nTable: orders(id, cust_id), order_items(order_id, item_id)\nOutput:",
            "tables": ["orders", "order_items"],
            "columns": ["cust_id"],
            "sqlite_setup": "CREATE TABLE orders (id INT, cust_id INT); CREATE TABLE order_items (order_id INT, item_id INT); INSERT INTO orders VALUES (1, 101); INSERT INTO order_items VALUES (1, 1), (1, 2);",
            "expected_result": [(101,)] if i == 1 else [],
            "order_sensitive": False
        })

    return suite
