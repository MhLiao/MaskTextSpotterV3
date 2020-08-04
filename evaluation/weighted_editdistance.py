def weighted_edit_distance(word1, word2, scores):
    m = len(word1)
    n = len(word2)
    dp = [[0 for __ in range(m + 1)] for __ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):  ## word2
        for j in range(1, m + 1): ## word1
            delect_cost = ed_delect_cost(j-1, i-1, word1, word2, scores)  ## delect a[i]
            insert_cost = ed_insert_cost(j-1, i-1, word1, word2, scores)  ## insert b[j]
            if word1[j - 1] != word2[i - 1]:
                replace_cost = ed_replace_cost(j-1, i-1, word1, word2, scores) ## replace a[i] with b[j]
            else:
                replace_cost = 0
            dp[i][j] = min(dp[i-1][j] + insert_cost, dp[i][j-1] + delect_cost, dp[i-1][j-1] + replace_cost)

    return dp[n][m]

def ed_delect_cost(j, i, word1, word2, scores):
    ## delect a[i]
    c = char2num(word1[j])
    return scores[c][j]


def ed_insert_cost(i, j, word1, word2, scores):
    ## insert b[j]
    if i < len(word1) - 1:
        c1 = char2num(word1[i])
        c2 = char2num(word1[i + 1])
        return (scores[c1][i] + scores[c2][i+1])/2
    else:
        c1 = char2num(word1[i])
        return scores[c1][i]


def ed_replace_cost(i, j, word1, word2, scores):
    ## replace a[i] with b[j]
    c1 = char2num(word1[i])
    c2 = char2num(word2[j])
    # if word1 == "eeatpisaababarait".upper():
    #     print(scores[c2][i]/scores[c1][i])

    return max(1 - scores[c2][i]/scores[c1][i]*5, 0)

def char2num(char):
    if char in '0123456789':
        num = ord(char) - ord('0') + 1
    elif char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        num = ord(char.lower()) - ord('a') + 11
    else:
        print('error symbol', char)
        exit()
    return num - 1