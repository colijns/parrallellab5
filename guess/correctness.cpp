#include "PCFG.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;

#ifndef VERIFY_MODE
#define VERIFY_MODE 1
// 1 = verify OpenMP
// 2 = verify pthread
#endif

#ifndef VERIFY_STEPS
#define VERIFY_STEPS 30
#endif

#ifndef TRAIN_PATH
#define TRAIN_PATH "guessdata/Rockyou-singleLined-full.txt"
#endif

static vector<string> CollectBufferedGuesses(PriorityQueue &q)
{
    vector<string> res;
    size_t total = q.guesses.size();

    for (const auto &bucket : q.local_guesses)
    {
        total += bucket.size();
    }

    res.reserve(total);
    res.insert(res.end(), q.guesses.begin(), q.guesses.end());

    for (const auto &bucket : q.local_guesses)
    {
        res.insert(res.end(), bucket.begin(), bucket.end());
    }

    return res;
}

static void ExpandFrontOnly(PriorityQueue &q)
{
    vector<PT> new_pts = q.priority.front().NewPTs();

    for (PT pt : new_pts)
    {
        q.CalProb(pt);

        for (auto iter = q.priority.begin(); iter != q.priority.end(); iter++)
        {
            if (iter != q.priority.end() - 1 && iter != q.priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    q.priority.emplace(iter + 1, pt);
                    break;
                }
            }

            if (iter == q.priority.end() - 1)
            {
                q.priority.emplace_back(pt);
                break;
            }

            if (iter == q.priority.begin() && iter->prob < pt.prob)
            {
                q.priority.emplace(iter, pt);
                break;
            }
        }
    }

    q.priority.erase(q.priority.begin());
}

static unsigned long long HashStrings(const vector<string> &v)
{
    unsigned long long h = 1469598103934665603ULL;

    for (const string &s : v)
    {
        for (unsigned char c : s)
        {
            h ^= c;
            h *= 1099511628211ULL;
        }

        h ^= 131;
        h *= 1099511628211ULL;
    }

    return h;
}

static bool CompareGuessSet(vector<string> serial_guesses,
                            vector<string> parallel_guesses,
                            int step)
{
    if (serial_guesses.size() != parallel_guesses.size())
    {
        cout << "[FAIL] Step " << step << ": size mismatch. "
             << "serial=" << serial_guesses.size()
             << ", parallel=" << parallel_guesses.size() << endl;
        return false;
    }

    sort(serial_guesses.begin(), serial_guesses.end());
    sort(parallel_guesses.begin(), parallel_guesses.end());

    for (size_t i = 0; i < serial_guesses.size(); i++)
    {
        if (serial_guesses[i] != parallel_guesses[i])
        {
            cout << "[FAIL] Step " << step << ": content mismatch at index "
                 << i << endl;
            cout << "serial  : " << serial_guesses[i] << endl;
            cout << "parallel: " << parallel_guesses[i] << endl;
            return false;
        }
    }

    cout << "[PASS] Step " << step
         << ", generated=" << serial_guesses.size()
         << ", checksum=" << HashStrings(serial_guesses)
         << endl;

    return true;
}

int main(int argc, char **argv)
{
    int steps = VERIFY_STEPS;

    if (argc >= 2)
    {
        steps = atoi(argv[1]);
        if (steps <= 0)
        {
            steps = VERIFY_STEPS;
        }
    }

    cout << "Correctness verification start." << endl;

#if VERIFY_MODE == 1
    cout << "Mode: serial Generate vs OpenMP Generate" << endl;
#elif VERIFY_MODE == 2
#if ENABLE_PTHREAD_GEN
    cout << "Mode: serial Generate vs pthread Generate" << endl;
#else
    cout << "VERIFY_MODE=2 requires ENABLE_PTHREAD_GEN=1." << endl;
    return 1;
#endif
#else
    cout << "Unknown VERIFY_MODE. Use 1 for OpenMP, 2 for pthread." << endl;
    return 1;
#endif

    cout << "Verify steps: " << steps << endl;

    PriorityQueue q_serial;
    PriorityQueue q_parallel;

    ifstream train_file(TRAIN_PATH);
    if (!train_file)
    {
        cerr << "Training file not found: " << TRAIN_PATH << endl;
        return 1;
    }
    train_file.close();

    cout << "Training serial model..." << endl;
    q_serial.m.train(TRAIN_PATH);
    q_serial.m.order();
    q_serial.init();

    cout << "Training parallel model..." << endl;
    q_parallel.m.train(TRAIN_PATH);
    q_parallel.m.order();
    q_parallel.init();

    long long total_serial = 0;
    long long total_parallel = 0;

    for (int step = 1; step <= steps; step++)
    {
        if (q_serial.priority.empty() || q_parallel.priority.empty())
        {
            cout << "Priority queue empty before step " << step << endl;
            break;
        }

        PT pt_serial = q_serial.priority.front();
        PT pt_parallel = q_parallel.priority.front();

        q_serial.ClearGuesses();
        q_parallel.ClearGuesses();

        q_serial.Generate(pt_serial);

#if VERIFY_MODE == 1
        q_parallel.Generate_openmp(pt_parallel);
#elif VERIFY_MODE == 2
#if ENABLE_PTHREAD_GEN
        q_parallel.Generate_pthread(pt_parallel);
#endif
#endif

        vector<string> serial_guesses = CollectBufferedGuesses(q_serial);
        vector<string> parallel_guesses = CollectBufferedGuesses(q_parallel);

        total_serial += (long long)serial_guesses.size();
        total_parallel += (long long)parallel_guesses.size();

        bool ok = CompareGuessSet(serial_guesses, parallel_guesses, step);

        if (!ok)
        {
            cout << "Correctness verification failed." << endl;
            return 1;
        }

        ExpandFrontOnly(q_serial);
        ExpandFrontOnly(q_parallel);
    }

    cout << "----------------------------------------" << endl;
    cout << "All checked steps passed." << endl;
    cout << "Total checked serial guesses  : " << total_serial << endl;
    cout << "Total checked parallel guesses: " << total_parallel << endl;
    cout << "Correctness verification PASS." << endl;

    return 0;
}
