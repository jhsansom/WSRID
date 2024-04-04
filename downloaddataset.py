import arxiv

# Construct the default API client.
client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = "ai",
  max_results = 500,
  sort_by = arxiv.SortCriterion.Relevance
)

results = client.results(search)

with open("sometext.txt", "w") as text_file:
  for r in client.results(search):
    print(r.title)
    print(r.summary)
    text_file.write(r.summary + '\n' + '\n')
  all_results = list(results)
  print(len(all_results))
  #print([r.title for r in all_results])
