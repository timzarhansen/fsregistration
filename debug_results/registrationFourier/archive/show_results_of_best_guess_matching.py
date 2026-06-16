import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import subprocess

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

file_name = "resultsOfManyMatching/gazeboCorrectedEvenAnglesPCLs_2_75ICP.mat"
data = sio.loadmat(file_name)

data2 = sio.loadmat("resultsOfManyMatching/gazeboCorrectedEvenAnglesPCLs_2_7564.mat")

resulting_yaw_diff_best_matching = data['resultingYawDiffBestMatching'].flatten()
resulting_yaw_diff_initial_guess = data['resultingYawDiffInitialGuess'].flatten()
resulting_yaw_diff = data['resultingYawDiff'].flatten()
translation_diff_best_matching = data['translationDiffBestMatching']
translation_diff_initial_guess = data['translationDiffInitialGuess']
translation_diff = data['translationDiff']

plt.figure(10)
plt.clf()
plt.hold(True)

last_idx_1 = np.where(resulting_yaw_diff_best_matching != 0)[0]
last_idx_2 = np.where(resulting_yaw_diff_initial_guess != 0)[0]
last_idx_3 = np.where(resulting_yaw_diff != 0)[0]

end_idx_1 = last_idx_1[-1] + 1 if len(last_idx_1) > 0 else len(resulting_yaw_diff_best_matching)
end_idx_2 = last_idx_2[-1] + 1 if len(last_idx_2) > 0 else len(resulting_yaw_diff_initial_guess)
end_idx_3 = last_idx_3[-1] + 1 if len(last_idx_3) > 0 else len(resulting_yaw_diff)

p1 = plt.plot(resulting_yaw_diff_best_matching[1:end_idx_1] * 180 / np.pi, alpha=0.75)
p2 = plt.plot(resulting_yaw_diff_initial_guess[1:end_idx_2] * 180 / np.pi, alpha=0.75)
p3 = plt.plot(resulting_yaw_diff[1:end_idx_3] * 180 / np.pi, alpha=0.75)

plt.xlabel("Scan Number")
plt.ylabel("Angle in Degree")
plt.gca().set_aspect('auto')

plt.legend(['Best Match FMS', 'Initial Guess FMS', 'Initial Guess ICP'], fontsize=10)

plt.savefig('outputPDFs/multipleScansMatchingAngleDiff.pdf', bbox_inches='tight', format='pdf')
subprocess.run(['pdfcrop', 'outputPDFs/multipleScansMatchingAngleDiff.pdf', 'outputPDFs/multipleScansMatchingAngleDiff.pdf'])

plt.figure(11)
plt.clf()
plt.hold(True)

translation_norm_best = np.linalg.norm(translation_diff_best_matching.T[1:end_idx_1, :], axis=1)
translation_norm_initial = np.linalg.norm(translation_diff_initial_guess.T[1:end_idx_2, :], axis=1)
translation_norm_icp = np.linalg.norm(translation_diff.T[1:81, :], axis=1)

p1 = plt.plot(translation_norm_best, alpha=0.75)
p2 = plt.plot(translation_norm_initial, alpha=0.75)
p3 = plt.plot(translation_norm_icp, alpha=0.75)

plt.xlabel("Scan Number")
plt.ylabel("Euclidean distance in m")
plt.gca().set_aspect('auto')

plt.legend(['Best Match FMS', 'Initial Guess FMS', 'Initial Guess ICP'], fontsize=10)

plt.savefig('outputPDFs/multipleScansMatchingTranslationDiff.pdf', bbox_inches='tight', format='pdf')
subprocess.run(['pdfcrop', 'outputPDFs/multipleScansMatchingTranslationDiff.pdf', 'outputPDFs/multipleScansMatchingTranslationDiff.pdf'])

mean_initial_guess_angle = np.mean(np.abs(resulting_yaw_diff_initial_guess[1:end_idx_1] * 180 / np.pi))
std_initial_guess_angle = np.std(np.abs(resulting_yaw_diff_initial_guess[1:end_idx_1] * 180 / np.pi))

mean_best_guess_angle = np.mean(np.abs(resulting_yaw_diff_best_matching[1:end_idx_1] * 180 / np.pi))
std_best_guess_angle = np.std(np.abs(resulting_yaw_diff_best_matching[1:end_idx_1] * 180 / np.pi))

mean_best_guess_translation = np.mean(translation_norm_best)
std_best_guess_translation = np.std(translation_norm_best)

mean_initial_guess_translation = np.mean(translation_norm_initial)
std_initial_guess_translation = np.std(translation_norm_initial)

mean_icp_angle = np.mean(np.abs(resulting_yaw_diff[1:end_idx_3] * 180 / np.pi))
std_icp_angle = np.std(np.abs(resulting_yaw_diff[1:end_idx_3] * 180 / np.pi))

mean_icp_translation = np.mean(translation_norm_icp)
std_icp_translation = np.std(translation_norm_icp)
