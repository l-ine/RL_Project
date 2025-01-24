import subprocess


def main():
    # m = [10, 20]
    n = [0.1, 0.3, 0.5]                   # eps
    # t = [32]                            # t
    l = [5e-05, 0.0001, 0.0005]           # l
    s = [1]#, 10, 20, 50, 100, 999]         # s
    a = ["DDPG-default", "pinkNoise", "pinkNoiseRND"]    # alg


    for a_element in a:
        for n_element in n:
            # for t_element in t:
            for l_element in l:
                for s_element in s:

                    # Command-Prompt
                    command = [
                        "python", "DDPG.py",
                        "-e", "Pendulum-v1",
                        "-alg", str(a_element),
                        "-n", str(n_element),
                        #"-t", str(t_element),
                        "-l", str(l_element),
                        # "-m", str(m_element),
                        "-s", str(s_element)
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
