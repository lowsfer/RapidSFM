/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include "FiberUtils.h"
#include "PriorityFiberPool.h"
#include "PipeLine.h"

#if 0
int main()
{
    const auto taskDuration = std::chrono::seconds{1};
    const int nbFibers = 100000;
    const int nbThreads = 4;

    if (true)
    {
        cudapp::FiberPool fiberPool{nbThreads};

        std::vector<fb::future<fb::future<void>>> fibers;
        for (int i = 0; i < nbFibers; i++) {
            auto f = fiberPool.async([taskDuration](){
                this_fiber::sleep_for(taskDuration);
                // std::this_thread::sleep_for(taskDuration);
            });
            fibers.emplace_back(std::move(f));
        }

        for (auto& f: fibers)
        {
            f.get();
        }
    }
    else
    {
        cudapp::FiberFactory fiberFactory{nbThreads};

        std::vector<cudapp::FiberFactory::FiberFuture> fibers;
        for (int i = 0; i < nbFibers; i++) {
            auto f = fiberFactory.create([taskDuration](){
                this_fiber::sleep_for(taskDuration);
                // std::this_thread::sleep_for(taskDuration);
            });
            fibers.emplace_back(std::move(f));
        }

        for (auto& f: fibers) {
            f.get().join();
        }
    }

    return 0;
}
#else

template <typename T>
using Channel = cudapp::IPipeLine<int, int>::Channel<T>;

int main()
{
    const int nbThreads = 12;
    cudapp::PriorityFiberPool fiberPool{nbThreads};

    Channel<int> products;
    const auto pass = [&fiberPool](int a){
        fiberPool.post([]{
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }).get();

        return a;
    };
    const auto terminal = [&fiberPool](int a){
        fiberPool.post([]{
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }).get();
        return a;
    };
    const auto pipeline = fiberPool.post([&]{
        return cudapp::makePipeLine([&fiberPool](int32_t priority, std::function<void()> f){return fiberPool.post(priority, f);},
            products, pass, pass, pass, pass, pass, terminal);
    }).get();

    const int nbTasks = 1<<20;
    for (int i = 0; i < 1; i++)
    {
        fiberPool.post([&]{
            for (int i = 0; i < nbTasks; i++)
            {
                pipeline->enqueue(i);
            }
        });

        while (true){
            std::stringstream ss;
            ss << "Pipeline Status:";
            for (int i = 0; i < static_cast<int>(pipeline->getNbStages()) + 1; i++) {
                ss << '\t' << pipeline->getCurrentChannelSize(i);
            }
            std::cout << ss.str() << std::endl;
            if (products.peekSize() == nbTasks){
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    return 0;
}

#endif
