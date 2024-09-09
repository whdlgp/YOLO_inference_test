#pragma once

#include <json.hpp>

// Helper for check JSON data is correct for input argument
template<typename T>
bool check_and_get(const nlohmann::json& input, const std::string& key, T& value) 
{
    if (input.contains(key) && input[key].is_primitive()) 
    {
        if (std::is_same_v<T, std::string> && input[key].is_string()) 
        {
            value = input[key];
            return true;
        } 
        else if (std::is_same_v<T, int> && input[key].is_number_integer()) 
        {
            value = input[key];
            return true;
        } 
        else if (std::is_same_v<T, bool> && input[key].is_boolean()) 
        {
            value = input[key];
            return true;
        }
        else if (std::is_same_v<T, float> && input[key].is_number_float()) 
        {
            value = input[key];
            return true;
        }
    }
    return false;
}