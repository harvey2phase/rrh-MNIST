import os
import datetime

def mkdir(path):
    if os.path.exists(path):
        print("Directory %s already exists!" % path)
    else:
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

def plot(x, x_recon, kind):
    count = 0
    if kind == "train":
        recon_c = "red"
    else:
        recon_c = "green"
    for i in range(1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.linspace(0, 127, 128), x.detach().cpu()[i], c = "blue")
        ax.scatter(np.linspace(0, 127, 128), x_recon.detach().cpu()[i], c = recon_c)
        #plt.show()
        plt.savefig(os.path.join(TEST, "training_curve.png"), dpi = 600)

def make_exp_folder(parent_folder, exp_name):
    exp_folder = os.path.join(parent_folder, exp_name)
    mkdir(exp_folder)
    exp_folder = os.path.join(
        parent_folder,
        exp_folder,
        datetime.datetime.now().strftime("%m-%d_%H-%M"),
    )
    mkdir(exp_folder)
    return exp_folder
