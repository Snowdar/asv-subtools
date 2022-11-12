//  Copyright xmuspeech (Author: Leo 2021-12-18)

#ifndef UTILS_OPTIONS_H
#define UTILS_OPTIONS_H

#include <map>
#include <string>
#include "yaml-cpp/yaml.h"

namespace subtools {

class Options{
    public:
        void Register(const std::string &name, bool *ptr);
        void Register(const std::string &name, int *ptr);
        void Register(const std::string &name, float *ptr);
        void Register(const std::string &name, double *ptr);
        void Register(const std::string &name, std::string *ptr);
        void ReadYamlMap(const YAML::Node &optm);

    private:
        template<typename T>
        void RegisterTmpl(const std::string &name, T *ptr);

        /// Register boolean variable
        void RegisterSpecific(const std::string &name, const std::string &idx,
                                bool *b);
        /// Register int variable
        void RegisterSpecific(const std::string &name, const std::string &idx,
                                int *i);
        /// Register float variable
        void RegisterSpecific(const std::string &name, const std::string &idx,
                                float *f);
        /// Register double variable [useful as we change BaseFloat type].
        void RegisterSpecific(const std::string &name, const std::string &idx,
                                double *f);
        /// Register string variable
        void RegisterSpecific(const std::string &name, const std::string &idx,
                                std::string *s);

        bool SetOptions(const std::string &key,const YAML::Node &value);
        // maps for option variables
        std::map<std::string, bool*> bool_map_;
        std::map<std::string, int*> int_map_;
        std::map<std::string, double*> double_map_;
        std::map<std::string, float*> float_map_;
        std::map<std::string, std::string*> string_map_;

    protected:
        void NormalizeArgName(std::string *str);
    };
}

#endif  // UTILS_OPTIONS_H
