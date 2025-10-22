using System.Diagnostics;
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
        int divisions = json.divisions;
        string className = json.className;
        string path = json.path;
        bool isMerge = json.isMerge;

        if (isMerge)
        {
            await mergeClasses(path, divisions, rest);
            return;
        }
        
        Process.Start( new ProcessStartInfo { FileName = "http://localhost:3000/collect", UseShellExecute = true });
        Thread.Sleep(5000);

        Directory.CreateDirectory(Path.Combine(path, className));
        
        const string url = "http://localhost:3000/mindwave/data";
        HttpClient client = new HttpClient();
        client.BaseAddress = new Uri(url);

        Object[,] largeJson = new Object[times, divisions];
        
        for (int i = 0; i < times; i++)
        {
            Object[] featuresList = await readSequence(divisions, rest, client, url);
            for (int j = 0; j < featuresList.Length; j++)
            {
                largeJson[i, j] = featuresList[j];
            }
        }
        string serializedJson = JsonConvert.SerializeObject(largeJson, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, className, $"{className}.json"), serializedJson);
    }

    async static Task<Object[]> readSequence(int divisions, int rest, HttpClient client, string url)
    {
        Object[] results = new Object[divisions];
        for (int i = 0; i < divisions; i++)
        {
            HttpResponseMessage response = await client.GetAsync(url);
            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine("This one didn't work..");
                break;
            }
            string responseString = await response.Content.ReadAsStringAsync();
            dynamic data = JsonConvert.DeserializeObject(responseString);
            results[i] = data["eeg"];
            Thread.Sleep(rest * 1000 / divisions);
        }
        return results;
    }

    async static Task mergeClasses(string path, int divisions, int rest)
    {
        if (!Directory.Exists(path))
        {
            Console.WriteLine("Directory doesn't exist");
            return;
        }

        string[] folders = Directory.GetDirectories(path);
        List<string> files = new List<string>();
        for (int i = 0; i < folders.Length; i++)
        {
            if (folders[i].Substring(folders[i].LastIndexOf("/") + 1) != "Formatted")
                files.Add(Path.Combine(path, folders[i],
                    folders[i].Substring(folders[i].LastIndexOf("/") + 1) + ".json"));
        }

        List<Object> objects = new List<Object>();
        for (int _label = 0; _label < files.Count; _label++)
        {
            dynamic json = JsonConvert.DeserializeObject(File.ReadAllText(files[_label]));

            if (divisions == 1)
            {
                foreach (dynamic jsonObject in json)
                {
                    dynamic jsonObj = jsonObject[0];
                    var payload = new
                    {
                        features = new Object[]
                        {
                            jsonObj.delta, jsonObj.theta, jsonObj.loAlpha, jsonObj.hiAlpha,
                            jsonObj.loBeta, jsonObj.hiBeta, jsonObj.loGamma, jsonObj.midGamma
                        },
                        label = _label
                    };
                    objects.Add(payload);
                }
            }
            else
            {
                foreach (dynamic jsonObject in json)
                {
                    List<Object[]> _features = new List<Object[]>();
                    foreach (dynamic jsonObj in jsonObject)
                    {
                        _features.Add(new Object[]
                        {
                            jsonObj.delta, jsonObj.theta, jsonObj.loAlpha, jsonObj.hiAlpha,
                            jsonObj.loBeta, jsonObj.hiBeta, jsonObj.loGamma, jsonObj.midGamma
                        });
                    }

                    var payload = new
                    {
                        features = _features,
                        label = _label
                    };
                    objects.Add(payload);
                }
            }
        }

        if (!Directory.Exists(Path.Combine(path, "Formatted")))
        {
            Directory.CreateDirectory(Path.Combine(path, "Formatted"));
        }

        string text = JsonConvert.SerializeObject(objects, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, "Formatted", $"Formatted.json"), text);
        
        Dictionary<string, string> key = new Dictionary<string, string>();
        for (int label = 0; label < files.Count; label++)
        {
            key.Add(label.ToString(), files[label].Substring(files[label].LastIndexOf("/") + 1, (files[label].Length - files[label].LastIndexOf("/")) - 6));
        }
        
        Dictionary<string, Object> entire = new Dictionary<string, Object>();
        entire.Add("classes", key);
        entire.Add("duration", rest);
        entire.Add("divisions", divisions);
        text = JsonConvert.SerializeObject(entire, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, "Formatted", $"Key.json"), text);
    }
}