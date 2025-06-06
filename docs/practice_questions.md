# Practice Questions and Key Concepts

Below are practice questions to solidify your understanding, followed by tables of key terms and fundamental propositions about architecture and the model registry.

## Practice Questions
1. What problem did the original flat model registry cause when multiple vendors provided the same model?
2. How does the new `_models_by_vendor` structure solve this issue?
3. Why do we keep the `available_models` dictionary, and what vendor does it prefer by default?
4. Imagine adding a new vendor tomorrow. What part of the design makes this simpler?
5. When might you still use `get_model()` without specifying a vendor?
6. Give a real life analogy that explains why storing vendor information separately helps.
7. What is a factory method in `AIModel` and why might we use it?
8. How does the registry pattern relate to a phone book or a library catalog?
9. Explain how the Façade pattern shows up in this code change.
10. What happens if you request a model that is not available for a specified vendor?

## Everyday Analogy
Think of a grocery store where the same fruit can come from different farms. If you only label bins by fruit type, you might lose track of which farm each piece came from. By having sub-bins for each farm, you can select oranges from Farm A or Farm B depending on your preference.

## Key Terms
| Category | Term | Proposition |
|----------|------|------------|
|Registry|Model Registry|"A registry is a type of catalog used to store and look up models"|
|Registry|Canonical Name|"A canonical name is a unique identifier for a model"|
|Registry|Vendor|"A vendor is a provider of a model"|
|Factory|Factory Method|"A factory method is a type of constructor for creating objects"|
|Factory|AIModel.claude_3_haiku|"`claude_3_haiku` is a factory method that returns an AIModel"|
|Compatibility|Backward Compatibility|"Backward compatibility is caused by the need to support older code"|
|Compatibility|Facade Pattern|"The facade pattern explains hiding complexity behind a simple interface"|
|Cost|ModelCosts|"ModelCosts is a data class that stores pricing information"|
|Cost|Cost Calculation|"Cost calculation is caused by multiplying token counts with token prices"|
|Resolution|Model Mapping|"Model mapping is a type of lookup between canonical names and vendor IDs"|
|Resolution|resolve_model_id|"`resolve_model_id` is a method that explains how to convert names to vendor-specific IDs"|
|Resolution|get_vendor_model_id|"`get_vendor_model_id` is a method that retrieves vendor-specific IDs"|
|Error Handling|ValueError|"ValueError is raised when invalid input causes an error"|
|Error Handling|Invalid Vendor|"An invalid vendor is a cause for raising an exception"|
|Testing|pytest|"pytest is a type of testing framework"|
|Testing|Unit Test|"A unit test explains how a small piece of code behaves"|
|Pattern|Registry Pattern|"Registry pattern is a type of design pattern"|
|Pattern|Factory Pattern|"Factory pattern is caused by needing consistent object creation"|
|Pattern|Mapping Table|"A mapping table explains associations between two domains"|
|Pattern|Two-Layer Cache|"Two-layer cache is a type of nested lookup design"|

## Fundamental Concepts
| Category | Concept | Proposition |
|----------|---------|-------------|
|Design|Separation of Concerns|"Separation of concerns is a type of modular design"|
|Design|Encapsulation|"Encapsulation explains hiding implementation details"|
|Design|Abstraction|"Abstraction is caused by representing complex ideas with simpler ones"|
|Data|Canonical Names|"Canonical names are a type of unique identifier"|
|Data|Vendor Variants|"Vendor variants are caused by different providers offering the same model"|
|Operations|Lookup|"Lookup explains retrieving a value from a registry"|
|Operations|Validation|"Validation is a type of error checking"|
|Operations|Mapping|"Mapping is caused by converting one identifier to another"|
|Evolution|Backward Compatibility|"Backward compatibility is a type of design constraint for evolution"|
|Evolution|Extensibility|"Extensibility explains how a system can grow with new vendors"|

