#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_events.h>

using namespace std;
class Square
{
    int x, y, pixel, borderSize, w, h;
    double *data;
    public:
    Square(int sX = 0, int sY = 0, int pixSize = 5, int sBorderSize = 0, int width = 0, int height = 0) : x(sX), y(sY), pixel(pixSize), borderSize(sBorderSize), w(width), h(height)
    {
        data = new double[w * h];
        if(data == NULL)
        {
            SDL_Log("Не удалось создать массив");
            return;
        }
    }
    void fill()
    {
        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }

        for(int i = 0; i < w * h; i++)
        {
            data[i] = 0;
        }
    }
    void changePos(int newX = 0, int newY = 0)
    {
        x = newX;
        y = newY;
    }
    void draw(SDL_Renderer *rend = nullptr)
    {
        // Border
        SDL_FRect rect = {x - pixel * w / 2.0 - borderSize, y - pixel * h / 2.0 - borderSize, pixel * w + borderSize * 2, pixel * h + borderSize * 2};
        if(!rend)
        {
            SDL_Log("Сцены не сущетсвует 0_0. Нулевой указатель на сцену");
            return;
        }
        SDL_RenderFillRect(rend, &rect);

        // Square content

        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }

        for(int i = 0; i < w * h; i++)
        {
            SDL_SetRenderDrawColor(rend, 255 * data[i], 255 * data[i], 255 * data[i], 0xFF);
            SDL_FRect rect = {x - pixel * (i % w - (w - 2) / 2.0), y - pixel * (i / w - (h - 2) / 2.0), pixel, pixel };
            SDL_RenderFillRect(rend, &rect);
        }
    }
    void click(double mouseX = 0, double mouseY = 0, bool paint = true)
    {
        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }
        int m = (x - mouseX + (w / 2.0) * pixel) / pixel;
        int n = (y - mouseY + (h / 2.0) * pixel) / pixel;
        if(m >= 0 & m < w)
        {
            if(n >= 0 & n < h)
            {
                if(paint)
                {
                    data[m + n * w] = 1.0;
                }
                else
                {
                    data[m + n * w] = 0.0;
                }
            }
        }  
    }
    
    void destroy()
    {
        delete[] data;
        data = nullptr;
    }
};
class Program
{
    protected:
    SDL_Window *window;
    SDL_Renderer *renderer;
    public:
    Program() : window(nullptr), renderer(nullptr) {};
    bool init(char *windowName, int w = 800, int h = 600, SDL_WindowFlags flags = 0)
    {
        if (!SDL_CreateWindowAndRenderer(windowName, w, h, flags, &window, &renderer)) 
        {
            SDL_Log("Не удалось создать окно: %s", SDL_GetError());
            return false;
        }

        SDL_Log("Используется рендеринг %s", SDL_GetRendererName(renderer));

        // Включение вертикальной синхронизации
        const int size = 4;
        int preffVsync[size] = {SDL_RENDERER_VSYNC_ADAPTIVE, 1, 2, SDL_RENDERER_VSYNC_DISABLED};
        bool vsync = false;
    
        for(int i = 0; i < size & !vsync; i++) // Выполнятется до тех пор пока не включится один из режимов вертикальной синхронизации
        {
            SDL_Log("Пробуем режим вертикальной синхронизации под номером %i", preffVsync[i]);
            vsync = SDL_SetRenderVSync(renderer, preffVsync[i]);
        }
        return true;
    }
    void quit()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};
class Intellect : public Program
{
    private:
    Square sq = {400, 300, 10, 10, 28, 28};
    public:
    bool init()
    {
        char title[] = "Шлюшны Інтылект";
        return Program::init(title, 800, 600, SDL_WINDOW_RESIZABLE);
        sq.fill();
    }
    void render()
    {
        // Black background
        SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
        SDL_RenderClear(renderer);

        // Rectangle

        SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0xAA, 0xFF);
        int w, h;
        SDL_GetWindowSizeInPixels(window, &w, &h);
        sq.changePos(w / 2, h / 2);
        sq.draw(renderer);
        

        SDL_RenderPresent(renderer);
    }
    void loop()
    {
        bool hold = false;
        int frames = 0;
        SDL_Time lastTime = SDL_GetTicks();
        while(true)
        {
            render();

            // Calculating FPS
            SDL_Time currTime = SDL_GetTicks();
            SDL_Time deltaTime = currTime - lastTime;
            frames++;
            if(deltaTime > 1000)
            {
                SDL_Log("FPS: %i", frames);
                lastTime = currTime;
                frames = 0;
            }


            SDL_Event e;
            while(SDL_PollEvent(&e))
            {
                if(e.type == SDL_EVENT_QUIT)
                {
                    quit();
                    return;
                } 
                else if(e.type == SDL_EVENT_MOUSE_MOTION & hold)
                {
                    if(e.button.button == 1) // Если левый клик - рисовать
                    {
                        sq.click(e.motion.x, e.motion.y, true);
                    }
                    else // Остальные клики - стирать
                    {
                        sq.click(e.motion.x, e.motion.y, false);
                    }
                }
                else if(e.type == SDL_EVENT_MOUSE_BUTTON_DOWN | e.type == SDL_EVENT_MOUSE_BUTTON_UP)
                {
                    hold = !hold;
                }
            }
        }
        
    }
    void quit()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        sq.destroy();
    }
};
int main()
{
    Intellect prog;
    prog.init();
    prog.loop();
    return 0;
}