#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

double function(double x, double y) 
{
    return sin(x + y) + sin(M_PI * (x + 2 * y)) + sin(sqrt(2) * (2 * x + y));
}

struct Point {
    double x;
    double y;
    double fitness;
};

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Определяем тип данных Point для MPI
    MPI_Datatype MPI_Point;
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_Point);
    MPI_Type_commit(&MPI_Point);

    double low_lim = -100.0;
    double high_lim = 100.0;

    random_device rd;
    mt19937 generator(rd() + rank);
    uniform_real_distribution<double> mutation_dist(-1.0, 1.0);
    uniform_real_distribution<double> for_y(-5000.0, 5000.0);

    int generations = 0;
    int population_size = 50000;
    double previous_max = 0.0; 
    double current_max = 0.0;

    double parallel_start_time = 0.0; // время начала параллельной части
    double parallel_end_time = 0.0; // время завершения параллельной части

    if (rank == 0)
    {
        parallel_start_time = MPI_Wtime(); // время начала параллельной части
    }

    vector<Point> local_population;

    for (int proc = rank; proc < 10; proc += size)
    {
        vector<Point> population(population_size);

        double begin_x = -5000.0 + proc * 1000.0;
        double end_x = -4000.0 + proc * 1000.0;

        uniform_real_distribution<double> for_x(begin_x, end_x);

        // Инициализация начальной популяции
        for (int i = 0; i < population_size; ++i)
        {
            double x = for_x(generator);
            double y = for_y(generator);
            double fitness = function(x, y);
            population[i] = { x, y, fitness };
        }

        // Сортировка популяции по значению fitness
        sort(population.begin(), population.end(), [](const Point& a, const Point& b) {return a.fitness > b.fitness;});

        previous_max = population[49].fitness;

        while (abs(previous_max - current_max) > 0.0001)
        {
            previous_max = population[49].fitness;

            // Мутация популяции
            for (int i = population_size / 2; i < population_size; ++i)
            {
                double new_x = population[i - population_size / 2].x + mutation_dist(generator);
                double new_y = population[i - population_size / 2].y + mutation_dist(generator);

                // Ограничение значений x и y в пределах сектора
                new_x = min(max(new_x, begin_x), end_x);
                new_y = min(max(new_y, -5000.0), 5000.0);

                double new_fitness = function(new_x, new_y);
                population[i] = { new_x, new_y, new_fitness };
            }

            // Сортировка популяции по значению fitness
            sort(population.begin(), population.end(), [](const Point& a, const Point& b) {return a.fitness > b.fitness;});
            current_max = population[49].fitness;

            generations++;
        }

        // Сокращение популяции до половины
        population.resize(50);
       
        for (int i = 0; i < population.size(); i++)
        {
            local_population.push_back(population[i]);
        }
    }

    // Сбор всех локальных популяций на нулевом процессе
    int local_size = local_population.size();
    vector<int> recv_counts(size);
    vector<int> displs(size);

    // Сбор размеров локальных популяций на нулевом процессе
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<Point> gathered_population;
    if (rank == 0) 
    {
        int total_size = 0;
        for (int i = 0; i < size; ++i) 
        {
            displs[i] = total_size;
            total_size += recv_counts[i];
        }
        gathered_population.resize(total_size);
    }

    // Сбор всех локальных популяций на нулевом процессе
    MPI_Gatherv(local_population.data(), local_size, MPI_Point, gathered_population.data(), recv_counts.data(), displs.data(), MPI_Point, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {  
        parallel_end_time = MPI_Wtime(); // окончание параллельной части
        cout << "Parallel Execution Time: " << parallel_end_time - parallel_start_time << " seconds\n"; // время параллельной части
   
        // Вывод лучших точек после всех поколений группами по 100 штук
        cout << "Collected points:" << endl;
        for (size_t i = 0; i < gathered_population.size(); i += 50) 
        {
            cout << "Group " << (i / 50) + 1 << ":" << endl;
            for (size_t j = i; j < i + 50 && j < gathered_population.size(); ++j) 
            {
                cout << "z(" << gathered_population[j].x << ", " << gathered_population[j].y << ") = " << gathered_population[j].fitness << endl;
            }
            cout << endl;
        }
        cout << endl;
    }

    // Освобождение пользовательского типа данных
    MPI_Type_free(&MPI_Point);
    MPI_Finalize();
    return 0;
}