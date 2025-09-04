using Newtonsoft.Json;

namespace Collecting;

using System.Threading;

class Program
{
    async static Task Main()
    {
        dynamic json = JsonConvert.DeserializeObject(File.ReadAllText("config.json"));
        int times = json.times;
        int rest = json.rest;
        string className = json.className;
        string path = json.path;
        
        Directory.CreateDirectory(Path.Combine(path, className));
        
        const string url = "http://localhost:3000/mindwave/data";
        HttpClient client = new HttpClient();
        client.BaseAddress = new Uri(url);

        Object[] largeJson = new Object[times];
        
        for (int i = 0; i < times; i++)
        {
            for (int j = 0; j < rest; j++)
            {
                Console.WriteLine("Get ready in " + (rest-j));
                Thread.Sleep(1000);
            }

            HttpResponseMessage response = await client.GetAsync(url);
            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine("This one didn't work..");
                break;
            }
            string responseString = await response.Content.ReadAsStringAsync();
            dynamic data = JsonConvert.DeserializeObject(responseString);
            largeJson[i] = data["eeg"];
            Console.WriteLine(data["eeg"]);
            Console.WriteLine("Completed");
        }
        string serializedJson = JsonConvert.SerializeObject(largeJson, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, className, $"{className}.json"), serializedJson);
    }
}