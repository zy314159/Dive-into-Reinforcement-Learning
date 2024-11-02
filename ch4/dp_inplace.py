V = {'L1': 0.0, 'L2': 0.0}

cnt = 0  # count the number of iterations
while True:
    temp = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    delta = abs(temp - V['L1'])
    V['L1'] = temp

    temp = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    delta = max(delta, abs(temp - V['L2']))
    V['L2'] = temp

    cnt += 1
    if delta < 1e-100:
        print(V)
        print(cnt)
        break