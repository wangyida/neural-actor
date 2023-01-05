import numpy as np
results = []
for f in ["metrics/actor1.txt", "metrics/actor2.txt","metrics/actor3.txt","metrics/actor4.txt","metrics/actor5.txt","metrics/actor6.txt","metrics/actor7.txt","metrics/actor8.txt",] :
    results.append(np.loadtxt(f))
results = np.asarray(results)
std = np.std(results, axis=0)
mean = np.mean(results, axis=0)
np.savetxt("metrics/mean.txt", mean)
np.savetxt("metrics/std.txt", std)
print('PSNR mean')
print(mean[:, 0].reshape((4, 5)))
print('PSNR std')
print(std[:, 0].reshape((4, 5)))
print('SSIM mean')
print(mean[:, 1].reshape((4, 5)))
print('SSIM std')
print(std[:, 1].reshape((4, 5)))
