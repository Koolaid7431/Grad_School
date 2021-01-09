
activity_names = {}
activity_names['A'] = 'Walking'
activity_names['B'] = 'Jogging'
activity_names['C'] = 'Stairs'
activity_names['D'] = 'Sitting'
activity_names['E'] = 'Standing'
activity_names['F'] = 'Typing'
activity_names['G'] = 'Brushing Teeth'
activity_names['H'] = 'Eating Soup'
activity_names['I'] = 'Eating Chips'
activity_names['J'] = 'Eating Pasta'
activity_names['K'] = 'Drinking from Cups'
activity_names['L'] = 'Eating Sandwich'
activity_names['M'] = 'Kicking (Soccer Ball)'
activity_names['O'] = 'Playing Catch w/Tennis Ball'
activity_names['P'] = 'Dribbling Basketball'
activity_names['Q'] = 'Writing'
activity_names['R'] = 'Clapping'
activity_names['S'] = 'Folding Clothes'

def plot3(x,y,z,fs,title): 
# x, y, and z are three direction of acceleration 
    # Calculate time values in seconds
    times = np.arange(x.shape[0], dtype='float') / fs

    plt.figure(figsize=(20, 8))
    plt.clf()
    plt.ylabel("Acceleration (m/s2) (OR gyroscop angular velocity (`/s))")
    plt.xlabel("Time (s)")
    plt.title(title)    
    plt.plot(times, x, "b", linewidth=0.8, label="X-direction")
    plt.legend(loc="lower right")

    plt.plot(times, y, "r", linewidth=0.8, label="Y-direction")
    plt.legend(loc="lower right")

    plt.plot(times, z, "k", linewidth=0.8, label="Z-direction")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()
    
def plot_5s(subj_activities,start_point):
    activity = np.asarray(subj_activities['A'],dtype=np.float64)
    x = activity[start_point:start_point+100,[0]]
    y = activity[start_point:start_point+100,[1]]
    z = activity[start_point:start_point+100,[2]]
    plot3(x,y,z,fs,'Walking')

    activity = np.asarray(subj_activities['B'],dtype=np.float64)
    x = activity[start_point:start_point+100,[0]]
    y = activity[start_point:start_point+100,[1]]
    z = activity[start_point:start_point+100,[2]]
    plot3(x,y,z,fs,'Jogging')

    activity = np.asarray(subj_activities['C'],dtype=np.float64)
    x = activity[start_point:start_point+100,[0]]
    y = activity[start_point:start_point+100,[1]]
    z = activity[start_point:start_point+100,[2]]
    plot3(x,y,z,fs,'Stairs')

    activity = np.asarray(subj_activities['D'],dtype=np.float64)
    x = activity[start_point:start_point+100,[0]]
    y = activity[start_point:start_point+100,[1]]
    z = activity[start_point:start_point+100,[2]]
    plot3(x,y,z,fs,'Sitting')

    activity = np.asarray(subj_activities['E'],dtype=np.float64)
    x = activity[start_point:start_point+100,[0]]
    y = activity[start_point:start_point+100,[1]]
    z = activity[start_point:start_point+100,[2]]
    plot3(x,y,z,fs,'Standing')
    
def plot_full_signal(subj_activities, action, activity_names):
    activity = np.asarray(subj_activities[action],dtype=np.float64)
    x = activity[0:1200,[0]]
    y = activity[0:1200,[1]]
    z = activity[0:1200,[2]]
    plot3(x,y,z,fs, activity_names[action]+'_0-60 seconds')

    x = activity[1200:2400,[0]]
    y = activity[1200:2400,[1]]
    z = activity[1200:2400,[2]]
    plot3(x,y,z,fs, activity_names[action]+'_60-120 seconds')

    x = activity[2400:,[0]]
    y = activity[2400:,[1]]
    z = activity[2400:,[2]]
    plot3(x,y,z,fs, activity_names[action]+'_120-end seconds')
