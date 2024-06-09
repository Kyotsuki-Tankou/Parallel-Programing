
/*
* void matMultiplyMPI(matCoo& x1, mat& x2, mat& res) {
    int n = x1.n, m = x2.m, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 定义自定义MPI数据类型，mat类传进去
    const int nitems = 3;
    int blocklengths[3] = { 1, 1, 1 };
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_INT, MPI_INT };
    MPI_Datatype mpi_elems_type;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(matCoo::elems, v);
    offsets[1] = offsetof(matCoo::elems, row);
    offsets[2] = offsetof(matCoo::elems, col);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_elems_type);
    MPI_Type_commit(&mpi_elems_type);

    if (rank == 0) {
        res.createMat(n, m);

        int rows_per_proc = n / p;
        int remaining_rows = n % p;

        for (int i = 1; i < p; i++) {
            int start_row = i * rows_per_proc + std::min(i, remaining_rows);
            int end_row = (i + 1) * rows_per_proc + std::min(i + 1, remaining_rows);
            int rows = end_row - start_row;
            MPI_Send(&rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&x1.elem[0], x1.totalElements, mpi_elems_type, i, 0, MPI_COMM_WORLD);
            MPI_Send(&x2.v[0][0], x2.n * x2.m, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&start_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            std::cout << "rank 0: Sent data to rank " << i << std::endl;
        }

        for (int i = 0; i < rows_per_proc + (remaining_rows > 0 ? 1 : 0); i++) {
            long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                int q = p;
                while (q < x1.elem.size() && x1.elem[q].row == i) {
                    sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                    q++;
                }
                res.v[i][j] = sum;
            }
        }

        for (int i = 1; i < p; i++) {
            int start_row = i * rows_per_proc + std::min(i, remaining_rows);
            int end_row = (i + 1) * rows_per_proc + std::min(i + 1, remaining_rows);
            int rows = end_row - start_row;
            std::cout << "rank 0: Ready to receive data from rank " << i << std::endl;
            MPI_Recv(&res.v[start_row][0], rows * m, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "rank 0: Received data from rank " << i << std::endl;
        }
        std::cout << "rank 0: End" << std::endl;
    }
    else {
        int rows;
        MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "rank " << rank << ": Received rows: " << rows << std::endl;

        matCoo x1_local;
        x1_local.createMat(x1.n, x1.m);
        x1_local.elem.resize(x1.totalElements);
        MPI_Recv(&x1_local.elem[0], x1.totalElements, mpi_elems_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "rank " << rank << ": Received x1" << std::endl;

        mat x2_local;
        x2_local.createMat(x2.n, x2.m);
        MPI_Recv(&x2_local.v[0][0], x2.n * x2.m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "rank " << rank << ": Received x2" << std::endl;

        int start_row;
        MPI_Recv(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "rank " << rank << ": Received start_row: " << start_row << std::endl;

        mat res_local;
        res_local.createMat(rows, m);

        for (int i = 0; i < rows; i++) {
            long long p = std::lower_bound(x1_local.elem.begin(), x1_local.elem.end(), start_row + i) - x1_local.elem.begin();
            if (p % 100 == 0)  std::cout << "rank " << rank << "p = " << p << std::endl;
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                int q = p;
                while (q < x1_local.elem.size() && x1_local.elem[q].row == start_row + i) {
                    sum += x1_local.elem[q].v * x2_local.v[x1_local.elem[q].col][j];
                    q++;
                }
                res_local.v[i][j] = sum;
                if (i % 100 == 0)  std::cout << "rank " << rank << "i = " << i << "j = " << j << " v[i][j] = " << res_local.v[i][j] << std::endl;
            }
        }
        std::cout << "rank " << rank << ": Processed data" << std::endl;
        MPI_Send(&res_local.v[0][0], rows * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        std::cout << "rank " << rank << ": Sent result" << std::endl;
    }

    MPI_Type_free(&mpi_elems_type);
}

*/