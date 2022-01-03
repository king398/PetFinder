import petpy
pf = petpy.Petfinder(key="gt1zwA9pFabcym3cGnCBaVkLPSRcOhL3IoI814lOH6ZjIHr7IN", secret="jUfeKOk7hD5uCPWGUJEOd1bB4F98ExepcDIdo6dI")
# Getting first 20 results without any search criteria

animals = pf.animals()

# Extracting data on specific animals with animal_ids

animal_ids = []
for i in animals['animals'][0:3]:
	animal_ids.append(i['id'])

animal_data = pf.animals(animal_id=animal_ids)

# Returning a pandas DataFrame of the first 150 animal results
animals = pf.animals(results_per_page=50, pages=3, return_df=True)
animals.head()