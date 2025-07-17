use vstd::prelude::*;

verus! {
    // A simplified CarPark that just tracks counts
    struct CarPark {
        weekend: bool,
        subscription_count: usize,
        car_park_count: usize,
        reserved_car_park_count: usize,
    }

    impl CarPark {
        fn new() -> (result: Self)
            ensures 
                result.subscription_count == 0,
                result.car_park_count == 0,
                result.reserved_car_park_count == 0,
                result.weekend == false,
        {
            CarPark {
                weekend: false,
                subscription_count: 0,
                car_park_count: 0,
                reserved_car_park_count: 0,
            }
        }

        fn leave_car_park(&mut self, _car: String) -> (success: bool)
            ensures 
                self.subscription_count == old(self).subscription_count && self.weekend == old(self).weekend,
        {
            if self.car_park_count > 0 {
                self.car_park_count = self.car_park_count - 1;
                true
            } else if self.reserved_car_park_count > 0 {
                self.reserved_car_park_count = self.reserved_car_park_count - 1;
                true
            } else {
                false
            }
        }

        fn check_availability(&self) -> (available_spaces: i32)
        {
            let normal_available = 2 - self.car_park_count as i32;
            let reserved_available = 3 - self.reserved_car_park_count as i32;
            
            if self.weekend {
                normal_available + reserved_available
            } else {
                normal_available
            }
        }

        fn make_subscription(&mut self, _car: String) -> (success: bool)
            ensures 
                self.car_park_count == old(self).car_park_count && self.reserved_car_park_count == old(self).reserved_car_park_count && self.weekend == old(self).weekend,
        {
            if self.subscription_count >= 3 {
                false
            } else {
                self.subscription_count = self.subscription_count + 1;
                true
            }
        }

        fn open_reserved_area(&mut self)
            ensures 
                self.car_park_count == old(self).car_park_count && self.reserved_car_park_count == old(self).reserved_car_park_count && 
                self.weekend == true && self.subscription_count == old(self).subscription_count,
        {
            self.weekend = true;
        }

        fn close_car_park(&mut self)
            ensures 
                self.car_park_count == 0 && self.reserved_car_park_count == 0 && self.subscription_count == 0,
                self.weekend == old(self).weekend,
        {
            self.car_park_count = 0;
            self.reserved_car_park_count = 0;
            self.subscription_count = 0;
        }

        fn enter_car_park(&mut self, _car: String) -> (success: bool)
            ensures 
                self.subscription_count == old(self).subscription_count && self.reserved_car_park_count == old(self).reserved_car_park_count && self.weekend == old(self).weekend,
        {
            if self.car_park_count >= 2 {
                false
            } else {
                self.car_park_count = self.car_park_count + 1;
                true
            }
        }

        fn enter_reserved_car_park(&mut self, _car: String) -> (success: bool)
            ensures 
                self.subscription_count == old(self).subscription_count && self.car_park_count == old(self).car_park_count && self.weekend == old(self).weekend,
        {
            if self.reserved_car_park_count >= 3 || (!self.weekend && self.subscription_count == 0) {
                false
            } else {
                self.reserved_car_park_count = self.reserved_car_park_count + 1;
                true
            }
        }
    }

    // Main function
    fn main() {
        let mut car_park = CarPark::new();

        // Test basic functionality
        let available_spaces = car_park.check_availability();
        assert(available_spaces == 2);

        let success = car_park.enter_car_park("car1".to_string());
        assert(success);
        assert(car_park.car_park_count == 1);

        let success = car_park.enter_car_park("car2".to_string());
        assert(success);
        assert(car_park.car_park_count == 2);

        let success = car_park.enter_car_park("car3".to_string());
        assert(!success);

        let success = car_park.make_subscription("car4".to_string());
        assert(success);
        assert(car_park.subscription_count == 1);

        let success = car_park.enter_reserved_car_park("car4".to_string());
        assert(success);
        assert(car_park.reserved_car_park_count == 1);

        let success = car_park.leave_car_park("car1".to_string());
        assert(success);
        assert(car_park.car_park_count == 1);

        car_park.close_car_park();
        assert(car_park.car_park_count == 0);
        assert(car_park.reserved_car_park_count == 0);
        assert(car_park.subscription_count == 0);
    }

    fn main_b() {
        let mut car_park = CarPark::new();

        assert(car_park.weekend == false);
        car_park.open_reserved_area();
        assert(car_park.weekend == true);

        let success = car_park.enter_reserved_car_park("car3".to_string());
        assert(success);
        assert(car_park.reserved_car_park_count == 1);

        car_park.close_car_park();
        assert(car_park.car_park_count == 0);
        assert(car_park.reserved_car_park_count == 0);
        assert(car_park.subscription_count == 0);
    }
}