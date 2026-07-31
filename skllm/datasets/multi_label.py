def get_multilabel_classification_dataset() -> tuple[list[str], list[list[str]]]:
    X = [
    "The product was of excellent quality, and the packaging was also very good. Highly recommend!",
    "The delivery was super fast, but the product did not match the information provided on the website.",
    "Great variety of products, but the customer support was quite unresponsive.",
    "Affordable prices and an easy-to-use website. A great shopping experience overall.",
    "The delivery was delayed, and the packaging was damaged. Not a good experience.",
    "Excellent customer support, but the return policy is quite complicated.",
    "The product was not as described. However, the return process was easy and quick.",
    "Great service and fast delivery. The product was also of high quality.",
    "The prices are a bit high. However, the product quality and user experience are worth it.",
    "The website provides detailed information about products. The delivery was also very fast."
    ]

    y = [
        ["Quality", "Packaging"],
        ["Delivery", "Product Information"],
        ["Product Variety", "Customer Support"],
        ["Price", "User Experience"],
        ["Delivery", "Packaging"],
        ["Customer Support", "Return Policy"],
        ["Product Information", "Return Policy"],
        ["Service", "Delivery", "Quality"],
        ["Price", "Quality", "User Experience"],
        ["Product Information", "Delivery"],
    ]

    return X, y
