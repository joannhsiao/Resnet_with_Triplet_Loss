import numpy as np
import matplotlib.pyplot as plt

def calc_counts(dis_list):
    np_row = np.array(dis_list).astype(np.float)
    count, bins_count = np.histogram(np_row, bins=100)
    return count, bins_count

def read_csv(desire_epoch = 1):
	pos_dis = []
	neg_dis = []
	i = 0
	with open("cosine_similarity.csv", "r") as csvfile:
	    data = csvfile.readlines()

	for i, line in enumerate(data):
	    if i > (len(data) - int(len(data) / desire_epoch)):
	        line = line.split(' ')
	        pos_dis.append(float(line[0]))
	        neg_dis.append(float(line[1]))
	return pos_dis, neg_dis

def calc_lines(pos_dis, neg_dis):
	pos_count, pos_bins_counts = calc_counts(pos_dis)
	neg_count, neg_bins_counts = calc_counts(neg_dis)

	# finding the PDF of the histogram using count values
	pos_line = pos_count / sum(pos_count)
	neg_line = neg_count / sum(neg_count)

	# using numpy np.cumsum to calculate the CDF
	pos_cdf = np.cumsum(pos_line)
	neg_cdf = 1 - np.cumsum(neg_line)
	return pos_line, neg_line, pos_cdf, neg_cdf, pos_bins_counts, neg_bins_counts

def draw_figures(pos_line, neg_line, pos_cdf, neg_cdf, pos_bins_counts, neg_bins_counts):
	plt.subplot(1, 2, 1) # row 1, col 2 index 1
	plt.title("Cosine distance")
	plt.xlabel('cosine distance')
	plt.plot(pos_bins_counts[1:], pos_line, color="red", label="pos distance")
	plt.plot(neg_bins_counts[1:], neg_line, label="neg distance")
	plt.legend()

	plt.subplot(1, 2, 2) # index 2
	plt.title("CDF")
	plt.xlabel('cosine distance')
	plt.plot(pos_bins_counts[1:], pos_cdf, color="red", label="pos cdf")
	plt.plot(neg_bins_counts[1:], neg_cdf, label="neg cdf")
	plt.legend()
	#plt.show()
	plt.savefig('./{}.jpg'.format("cosine_similarity_cdf"))

if __name__ == '__main__':
    # Enter which epoch do you wanna graph below
	desire_epoch = 6
	pos_dis, neg_dis = read_csv(desire_epoch = desire_epoch)
	pos_line, neg_line, pos_cdf, neg_cdf, pos_bins_counts, neg_bins_counts = calc_lines(pos_dis = pos_dis, neg_dis = neg_dis)
	draw_figures(pos_line, neg_line, pos_cdf, neg_cdf, pos_bins_counts, neg_bins_counts)
	

	

