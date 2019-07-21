a = [['97310/1563701397.3489714.jpg'],
['97310/1563701398.4822788.jpg'],
['97310/1563701399.6412935.jpg', 0.93449378013611],
['97310/1563701400.994029.jpg', 0.5693449378013611]]

first_filter = []
filter1_image_name = []
filter1_confidence_score = []
discarding_filter = []

for x in a:
	if(len(x) == 2):
		filter1_image_name.append(x[0])
		filter1_confidence_score.append(x[1])
	else:
		discarding_filter.append(x[0])
			

# print(filter1)
# print(discarding_filter)

max_conf_index = filter1_confidence_score.index(max(filter1_confidence_score))

max_conf_index_img = filter1_image_name[max_conf_index]

# for j in filter1_confidence_score:
# 	print(j)
# values.index(min(values))