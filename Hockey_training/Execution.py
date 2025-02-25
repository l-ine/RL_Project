import subprocess

# script for testing different parameters for DDPG and TD3
def main():
    m = 2500
    n = [0.1, 0.3, 0.5]                   # eps
    u = [10, 20, 50, 100]                 # u
    l = 0.0005 #[5e-05, 0.0001, 0.0005]           # l
    s = 1 #, 10, 20, 50, 100, 999]         # s
    a = ["DDPG-default", "pinkNoise", "RND"]    # alg

    for a_element in a:
        for n_element in n:
        #for u_element in u:
            #for l_element in l:

            # Command-Prompt
            command = [
                "python", "training.py",
                "-e", "Hockey",
                #"-alg", str(a_element),
                "-n", str(n_element),
                "-u", str(u),
                "-l", str(l),
                "-m", str(m),
                "-s", str(s)
            ]

            # Execute the prompt
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print("Output:")
                print(result.stdout)
                if result.stderr:
                    print("Error:")
                    print(result.stderr)
            except subprocess.CalledProcessError as e:
                print("Error when executing the command:")
                print(e.stderr)


if __name__ == '__main__':
    main()
