from ddgs import DDGS

with DDGS() as ddgs:
    results = list(
        ddgs.text(
            "Prime Minister Narendra Modi",
            max_results=5
        )
    )

for result in results:
    print(result)