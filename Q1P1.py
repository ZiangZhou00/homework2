
from roc import*
from data_generation import*

pdf = {
    'priors': np.array([0.6, 0.4]),
    'gmm_a': np.array([0.5, 0.5, 0.5, 0.5]),
    'mu': np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]]),
    'Sigma': np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]]
    ])
}

N_train = [20, 200, 2000]
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

data = [generate_q1_data(N, pdf) for N in N_train]
X_train, labels_train = zip(*data)

N_labels_train = np.array([(labels == 0).sum() for labels in labels_train]), np.array([(labels == 1).sum() for labels in labels_train])

for i, ax_i in enumerate(ax.flatten()[:-1]):
    ax_i.set_title(r"Training $D^{%d}$" % (N_train[i]))
    ax_i.plot(X_train[i][labels_train[i] == 0, 0], X_train[i][labels_train[i] == 0, 1], 'ro', label="Class 0")
    ax_i.plot(X_train[i][labels_train[i] == 1, 0], X_train[i][labels_train[i] == 1, 1], 'g+', label="Class 1")
    ax_i.set_xlabel(r"$x_1$")
    ax_i.set_ylabel(r"$x_2$")
    ax_i.legend()

N_valid = 10000

X_valid, labels_valid = generate_q1_data(N_valid, pdf)

Nl_valid = np.array([(labels_valid == 0).sum(), (labels_valid == 1).sum()])

ax[1, 1].set_title(r"Validation $D^{%d}$" % (N_valid))
ax[1, 1].plot(X_valid[labels_valid == 0, 0], X_valid[labels_valid == 0, 1], 'ro', label="Class 0")
ax[1, 1].plot(X_valid[labels_valid == 1, 0], X_valid[labels_valid == 1, 1], 'g+', label="Class 1")
ax[1, 1].set_xlabel(r"$x_1$")
ax[1, 1].set_ylabel(r"$x_2$")
ax[1, 1].legend()

x1_valid_lim = (np.floor(X_valid[:, 0].min()), np.ceil(X_valid[:, 0].max()))
x2_valid_lim = (np.floor(X_valid[:, 1].min()), np.ceil(X_valid[:, 1].max()))
plt.setp(ax, xlim=x1_valid_lim, ylim=x2_valid_lim)
plt.tight_layout()
plt.show()


def compute_discriminant_scores(X, dist_params):
    # Compute the discriminant scores of each sample in X
    class_lld_0 = (dist_params['gmm_a'][0] * mvn.pdf(X, dist_params['mu'][0], dist_params['Sigma'][0])
                   + dist_params['gmm_a'][1] * mvn.pdf(X, dist_params['mu'][1], dist_params['Sigma'][1]))
    class_lld_1 = mvn.pdf(X, dist_params['mu'][2], dist_params['Sigma'][2])
    discriminant_scores = np.log(class_lld_1) - np.log(class_lld_0)
    return discriminant_scores

# Compute the discriminant scores of the validation set using the PDF parameters
disc_scores_valid = compute_discriminant_scores(X_valid, pdf)

# Construct the ROC curve for the ERM classifier
roc_erm, gammas_empirical = estimate_roc(disc_scores_valid, labels_valid, Nl_valid)

# Plot the ROC curve
fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
ax_roc.plot(roc_erm['p10'], roc_erm['p11'], label="Empirical ERM Classifier ROC Curve")
ax_roc.set_xlabel(r"Probability of False Alarm $p(D=1\,|\,L=0)$")
ax_roc.set_ylabel(r"Probability of True Positive $p(D=1\,|\,L=1)$")

# Compute the empirical probability of error for each threshold
prob_error_empirical = np.array((roc_erm['p10'], 1 - roc_erm['p11'])).T.dot(Nl_valid / N_valid)

# Find the empirical threshold that minimizes the probability of error
min_prob_error_empirical = np.min(prob_error_empirical)
min_ind_empirical = np.argmin(prob_error_empirical)

# Compute the theoretical threshold that minimizes the probability of error (using the MAP rule)
gamma_map = pdf['priors'][0] / pdf['priors'][1]
decisions_map = disc_scores_valid >= np.log(gamma_map)
class_metrics_map = get_binary_classification_metrics(decisions_map, labels_valid, Nl_valid)
min_prob_error_map = np.array((class_metrics_map['FPR'] * pdf['priors'][0] + class_metrics_map['FNR'] * pdf['priors'][1]))

# Plot the empirical and theoretical thresholds
ax_roc.plot(roc_erm['p10'][min_ind_empirical], roc_erm['p11'][min_ind_empirical], 'go',
            label="Empirical Min Pr(error) ERM", markersize=14)
ax_roc.plot(class_metrics_map['FPR'], class_metrics_map['TPR'], 'rx',
            label="Theoretical Min Pr(error) ERM", markersize=14)

plt.grid(True)
plt.legend()
plt.show()

# Print the minimum probability of error and the corresponding threshold for the empirical and theoretical cases
print("Empirical: Min Pr(error) = {:.4f}, Min Gamma = {:.3f}".format(min_prob_error_empirical,
                                                                      np.exp(gammas_empirical[min_ind_empirical])))
print("Theoretical: Min Pr(error) = {:.4f}, Min Gamma = {:.3f}".format(min_prob_error_map, gamma_map))


# Compute the decision boundary using the discriminant scores and gamma
gamma = np.exp(gammas_empirical[min_ind_empirical])
decisions = disc_scores_valid >= np.log(gamma)

# Separate correctly and incorrectly classified points
correct_class_0 = X_valid[(decisions == 0) & (labels_valid == 0)]
correct_class_1 = X_valid[(decisions == 1) & (labels_valid == 1)]
incorrect_class_0 = X_valid[(decisions == 1) & (labels_valid == 0)]
incorrect_class_1 = X_valid[(decisions == 0) & (labels_valid == 1)]

# Plot the decision boundary and the validation dataset
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(correct_class_0[:, 0], correct_class_0[:, 1], c='g', label='Correctly classified 0')
ax.scatter(correct_class_1[:, 0], correct_class_1[:, 1], c='b', label='Correctly classified 1')
ax.scatter(incorrect_class_0[:, 0], incorrect_class_0[:, 1], c='r', label='Incorrectly classified 0')
ax.scatter(incorrect_class_1[:, 0], incorrect_class_1[:, 1], c='m', label='Incorrectly classified 1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Boundary Decision ')
plt.legend()
plt.show()
# Reference from MARK ZOLOTAS






