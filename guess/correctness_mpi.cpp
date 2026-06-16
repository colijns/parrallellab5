#include "PCFG.h"
#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

#ifndef VERIFY_STEPS
#define VERIFY_STEPS 30
#endif

static const string TRAIN_PATH = "/guessdata/Rockyou-singleLined-full.txt";

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

static vector<string> GatherStringsToRoot(const vector<string> &local,
                                          int rank,
                                          int world_size)
{
    int local_count = (int)local.size();
    vector<int> string_counts;
    if (rank == 0)
    {
        string_counts.resize(world_size);
    }

    MPI_Gather(&local_count, 1, MPI_INT,
               rank == 0 ? string_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    vector<int> local_lengths;
    local_lengths.reserve(local.size());

    int local_bytes = 0;
    for (const string &s : local)
    {
        int len = (int)s.size();
        local_lengths.push_back(len);
        local_bytes += len;
    }

    vector<int> length_displs;
    vector<int> all_lengths;
    if (rank == 0)
    {
        length_displs.resize(world_size);
        int total_strings = 0;
        for (int i = 0; i < world_size; i++)
        {
            length_displs[i] = total_strings;
            total_strings += string_counts[i];
        }
        all_lengths.resize(total_strings);
    }

    MPI_Gatherv(local_count > 0 ? local_lengths.data() : nullptr,
                local_count,
                MPI_INT,
                rank == 0 ? all_lengths.data() : nullptr,
                rank == 0 ? string_counts.data() : nullptr,
                rank == 0 ? length_displs.data() : nullptr,
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    string local_payload;
    local_payload.reserve(local_bytes);
    for (const string &s : local)
    {
        local_payload += s;
    }

    vector<int> byte_counts;
    if (rank == 0)
    {
        byte_counts.resize(world_size);
    }

    MPI_Gather(&local_bytes, 1, MPI_INT,
               rank == 0 ? byte_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    vector<int> byte_displs;
    vector<char> all_payload;
    if (rank == 0)
    {
        byte_displs.resize(world_size);
        int total_bytes = 0;
        for (int i = 0; i < world_size; i++)
        {
            byte_displs[i] = total_bytes;
            total_bytes += byte_counts[i];
        }
        all_payload.resize(total_bytes);
    }

    MPI_Gatherv(local_bytes > 0 ? &local_payload[0] : nullptr,
                local_bytes,
                MPI_CHAR,
                rank == 0 && !all_payload.empty() ? all_payload.data() : nullptr,
                rank == 0 ? byte_counts.data() : nullptr,
                rank == 0 ? byte_displs.data() : nullptr,
                MPI_CHAR,
                0,
                MPI_COMM_WORLD);

    vector<string> gathered;
    if (rank != 0)
    {
        return gathered;
    }

    int length_cursor = 0;
    for (int proc = 0; proc < world_size; proc++)
    {
        int payload_cursor = byte_displs[proc];

        for (int j = 0; j < string_counts[proc]; j++)
        {
            int len = all_lengths[length_cursor++];
            gathered.emplace_back(all_payload.data() + payload_cursor, len);
            payload_cursor += len;
        }
    }

    return gathered;
}

static bool CompareGuessSet(vector<string> serial_guesses,
                            vector<string> mpi_guesses,
                            int step,
                            long long mpi_count)
{
    long long serial_count = (long long)serial_guesses.size();
    if (serial_count != mpi_count || serial_guesses.size() != mpi_guesses.size())
    {
        cout << "[FAIL] Step " << step
             << ", serial_count=" << serial_count
             << ", mpi_count=" << mpi_count
             << ", gathered_count=" << mpi_guesses.size()
             << endl;
        return false;
    }

    sort(serial_guesses.begin(), serial_guesses.end());
    sort(mpi_guesses.begin(), mpi_guesses.end());

    for (size_t i = 0; i < serial_guesses.size(); i++)
    {
        if (serial_guesses[i] != mpi_guesses[i])
        {
            cout << "[FAIL] Step " << step
                 << ", serial_count=" << serial_count
                 << ", mpi_count=" << mpi_count
                 << ", mismatch_index=" << i
                 << endl;
            cout << "serial: " << serial_guesses[i] << endl;
            cout << "mpi   : " << mpi_guesses[i] << endl;
            return false;
        }
    }

    cout << "[PASS] Step " << step
         << ", serial_count=" << serial_count
         << ", mpi_count=" << mpi_count
         << ", checksum=" << HashStrings(serial_guesses)
         << endl;

    return true;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int steps = VERIFY_STEPS;
    if (argc >= 2)
    {
        steps = atoi(argv[1]);
        if (steps <= 0)
        {
            steps = VERIFY_STEPS;
        }
    }

    if (rank == 0)
    {
        cout << "MPI correctness verification start." << endl;
        cout << "Verify steps: " << steps << endl;
        cout << "MPI processes: " << world_size << endl;
    }

    streambuf *old_cout_buf = nullptr;
    ostringstream quiet_stream;
    if (rank != 0)
    {
        old_cout_buf = cout.rdbuf(quiet_stream.rdbuf());
    }

    PriorityQueue q_mpi;
    q_mpi.m.train(TRAIN_PATH);
    q_mpi.m.order();
    q_mpi.init();

    if (rank != 0)
    {
        cout.rdbuf(old_cout_buf);
    }

    PriorityQueue q_serial;
    if (rank == 0)
    {
        cout << "Training serial model..." << endl;
        q_serial.m.train(TRAIN_PATH);
        q_serial.m.order();
        q_serial.init();
    }

    for (int step = 1; step <= steps; step++)
    {
        if (q_mpi.priority.empty())
        {
            if (rank == 0)
            {
                cout << "Priority queue empty before step " << step << endl;
            }
            break;
        }

        q_mpi.ClearGuesses();
        PT pt_mpi = q_mpi.priority.front();

        vector<string> serial_guesses;
        if (rank == 0)
        {
            if (q_serial.priority.empty())
            {
                cout << "Serial priority queue empty before step " << step << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            q_serial.ClearGuesses();
            PT pt_serial = q_serial.priority.front();
            q_serial.Generate(pt_serial);
            serial_guesses = CollectBufferedGuesses(q_serial);
        }

        int pt_global_count = 0;
        q_mpi.Generate_mpi(pt_mpi, rank, world_size, &pt_global_count);
        vector<string> local_mpi_guesses = CollectBufferedGuesses(q_mpi);

        long long local_count = (long long)local_mpi_guesses.size();
        long long mpi_count = 0;
        MPI_Reduce(&local_count, &mpi_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        vector<string> gathered_mpi_guesses = GatherStringsToRoot(local_mpi_guesses, rank, world_size);

        int step_ok = 1;
        if (rank == 0)
        {
            if (mpi_count != (long long)pt_global_count)
            {
                cout << "[FAIL] Step " << step
                     << ", mpi_count=" << mpi_count
                     << ", pt_global_count=" << pt_global_count
                     << endl;
                step_ok = 0;
            }
            else if (!CompareGuessSet(serial_guesses, gathered_mpi_guesses, step, mpi_count))
            {
                step_ok = 0;
            }
        }

        MPI_Bcast(&step_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!step_ok)
        {
            MPI_Finalize();
            return 1;
        }

        ExpandFrontOnly(q_mpi);
        if (rank == 0)
        {
            ExpandFrontOnly(q_serial);
        }
    }

    if (rank == 0)
    {
        cout << "----------------------------------------" << endl;
        cout << "Correctness verification PASS." << endl;
    }

    MPI_Finalize();
    return 0;
}
