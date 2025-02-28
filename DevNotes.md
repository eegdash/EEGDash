## Install locally

pip install -r requirements.txt

## signalstore mongodb
- Check args functions to double check input to db query
- Create_index to the collection once its created to speed up querying
- `find` has deserialization to convert timestamp to correct milisecond format and json_schema from bytes to dict
- `add` has serialization before insert into db