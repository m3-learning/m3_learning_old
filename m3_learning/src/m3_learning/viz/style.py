def style(name = "default"):
    if name == "default":
        try:
            import seaborn as sns
            
            # resetting default seaborn style
            sns.reset_orig()
            
            print(f"{name} set for seaborn")
            
        except:
            pass
        
        try: 
            import matplotlib.pyplot as plt
            # setting default plotting params
            plt.rcParams['image.cmap'] = 'magma'
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['xtick.labelsize'] = 14
            plt.rcParams['ytick.labelsize'] = 14
            plt.rcParams['figure.titlesize'] = 20
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            print(f"{name} set for matplotlib")
        except:
            pass
            



