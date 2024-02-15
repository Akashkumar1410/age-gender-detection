class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price


class ShoppingCart:
    def __init__(self):
        self.cart = []

    def add_item(self, item):
        self.cart.append(item)

    def total_items(self):
        return len(self.cart)

    def total_cost(self):
        return sum(item.price for item in self.cart)


# Sample input
if __name__ == '__main__':
    items = [
        Item("bike", 1000),
        Item("headphones", 100)
    ]

    cart = ShoppingCart()

    operations = [
        cart.total_cost,
        lambda: cart.add_item(items[0]),
        cart.total_items,
        cart.total_cost,
        lambda: cart.add_item(items[1]),
        lambda: cart.add_item(items[1]),
        cart.total_items,
        cart.total_cost
    ]

    for operation in operations:
        result = operation()
        print(result)
