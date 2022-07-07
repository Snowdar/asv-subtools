//  Copyright xmuspeech (Author: Leo 2021-12-18)

#include "utils/options.h"
#include "glog/logging.h"

namespace subtools {
    void Options::Register(const std::string &name, bool *ptr)
    {
        RegisterTmpl(name, ptr);
    }

    void Options::Register(const std::string &name, int *ptr)
    {
        RegisterTmpl(name, ptr);
    }
    void Options::Register(const std::string &name, float *ptr)
    {
        RegisterTmpl(name, ptr);
    }
    void Options::Register(const std::string &name, double *ptr)
    {
        RegisterTmpl(name, ptr);
    }
    void Options::Register(const std::string &name, std::string *ptr)
    {
        RegisterTmpl(name, ptr);
    }

    template<typename T>
    void Options::RegisterTmpl(const std::string &name, T *ptr)
    {
        CHECK(ptr != NULL);
        std::string idx = name;
        NormalizeArgName(&idx);
        this->RegisterSpecific(name, idx, ptr);
    }

    void Options::RegisterSpecific(const std::string &name,
    const std::string &idx, bool *b)
    {
        bool_map_[idx] = b;
    }

    void Options::RegisterSpecific(const std::string &name,
    const std::string &idx, int *i)
    {
        int_map_[idx] = i;
    }

    void Options::RegisterSpecific(const std::string &name,
    const std::string &idx, double *f)
    {
        double_map_[idx] = f;
    }

    void Options::RegisterSpecific(const std::string &name,
    const std::string &idx, float *f)
    {
        float_map_[idx] = f;
    }

    void Options::RegisterSpecific(const std::string &name,
    const std::string &idx, std::string *s)
    {
        string_map_[idx] = s;
    }

    void Options::ReadYamlMap(const YAML::Node &optm)
    {
        CHECK(optm.IsMap());
        for (YAML::Node::const_iterator it = optm.begin(); it != optm.end(); ++it)
        {
            std::string key = it->first.as<std::string>();
            NormalizeArgName(&key);
            if(!SetOptions(key,it->second))
            {
                LOG(FATAL) << "Invalid option key: " << it->first << "value: " << it->second;
            }
        }
    }

    bool Options::SetOptions(const std::string &key,const YAML::Node &value)
    {
        CHECK(value.IsScalar());

        if (int_map_.end() != int_map_.find(key))
        {
            *(int_map_[key]) = value.as<int>();
        }
        else if(bool_map_.end() != bool_map_.find(key))
        {
            *(bool_map_[key]) = value.as<bool>();
        }
        else if(double_map_.end() != double_map_.find(key))
        {
            *(double_map_[key]) = value.as<double>();
        }
        else if(float_map_.end() != float_map_.find(key))
        {
            *(float_map_[key]) = value.as<float>();
        }
        else if(string_map_.end() != string_map_.find(key))
        {
            *(string_map_[key]) = value.as<std::string>();
        }
        else
        {
            return false;
        }

        return true;
    }

    void Options::NormalizeArgName(std::string *str)
    {
        std::string out;
        std::string::iterator it;

        for (it = str->begin(); it != str->end(); ++it)
        {
            if (*it == '_')
                out += '-';  // convert _ to -
            else
                out += std::tolower(*it);
        }
        *str = out;

        CHECK(str->length() > 0);
    }
}
